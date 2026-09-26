"""HTTP API + the built frontend. Everything the UI does goes through these routes (and so can a script)."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..core import imported, trace
from ..core.harness import HarnessError, list_demos, load_harness
from ..core.llm import LLMClient
from ..core.types import Endpoint, ExperimentConfig, HarnessRef, slug
from . import modelhub, state
from .manager import Manager, PreflightError

WEBUI = Path(__file__).resolve().parent.parent / "webui"


class ProbeReq(BaseModel):
    endpoint: Endpoint
    need_logprobs: bool = False


class TraceReq(BaseModel):
    """Either a path on this machine, or the file's text pasted/uploaded by the browser."""

    path: str | None = None
    text: str | None = None
    filename: str = "uploaded.jsonl"
    limit_tasks: int | None = None


class ProjectReq(TraceReq):
    accept: dict[str, str] = {}
    llm_price_in: float = 0.0
    llm_price_out: float = 0.0
    decider_price_in: float = 0.0
    decider_price_out: float = 0.0
    avg_prompt_tokens: int | None = None
    avg_output_tokens: int | None = None


class BuildReq(ProjectReq):
    name: str = "Imported pipeline"


class PullReq(BaseModel):
    repo_id: str


class ServeReq(BaseModel):
    model: str
    gpu_util: float = 0.15
    max_len: int = 8192
    max_seqs: int = 16
    port: int | None = None
    served_name: str | None = None


class StopReq(BaseModel):
    name: str


def create_app() -> FastAPI:
    app = FastAPI(title="JevControl", version=__version__)
    mgr = Manager()

    @app.exception_handler(PreflightError)
    async def _pre(_, e: PreflightError):
        return JSONResponse({"detail": str(e), "kind": "preflight"}, status_code=400)

    @app.get("/api/health")
    def health():
        ok, why = modelhub.docker_ok()
        return {"ok": True, "version": __version__, "docker": ok, "docker_note": why}

    # ---- endpoints ------------------------------------------------------------------------------------

    @app.post("/api/probe")
    def probe(req: ProbeReq):
        if req.endpoint.kind == "jev":  # a typed decision API: ask it one real question
            from ..core.jevapi import JevClient

            jc = JevClient(req.endpoint)
            try:
                r = jc.probe()
            finally:
                jc.close()
            return {"ok": r["ok"], "models": [], "chat_ok": r["ok"], "logprobs_ok": r["ok"],
                    "latency_ms": r.get("latency_ms", 0.0), "error": r.get("error", ""),
                    "model": req.endpoint.model, "sample_probs": r.get("sample_probs"),
                    "readout_ms": r.get("latency_ms"), "note": r.get("note", ""),
                    "label_mass": 1.0 if r["ok"] else None}
        c = LLMClient(req.endpoint)
        try:
            p = c.probe(req.need_logprobs)
            out = {"ok": p.ok, "models": p.models, "chat_ok": p.chat_ok, "logprobs_ok": p.logprobs_ok,
                   "latency_ms": p.latency_ms, "error": p.error, "model": req.endpoint.model}
            if p.ok and req.need_logprobs:  # is the model actually answering with menu letters?
                from ..core import menu

                r = menu.readout(c, "The customer wants a refund.", "choice", "Which team handles this?",
                                 ["billing", "shipping"], {"billing": "money, refunds", "shipping": "delivery"})
                out["label_mass"] = round(r.label_mass, 3)
                out["sample_probs"] = [round(x, 3) for x in r.probs]
                out["readout_ms"] = round(r.latency_ms, 1)
            return out
        finally:
            c.close()

    # ---- harness ----------------------------------------------------------------------------------------
    def describe(ref: HarnessRef) -> dict[str, Any]:
        h = load_harness(ref)
        tasks = h.load_tasks()
        return {"id": h.id, "name": h.name, "description": h.meta.get("description", ""), "sites": h.meta.get("sites", {}),
                "n_tasks": len(tasks), "sample_task": tasks[0], "has_score": h.score is not None,
                "has_truth": any("truth" in t for t in tasks), "tools": sorted(h.tools),
                "path": str(h.path), "tasks_path": str(h.tasks_path),
                "kinds": _count(t.get("kind", "") for t in tasks)}

    def _count(it):
        out: dict[str, int] = {}
        for k in it:
            if k:
                out[k] = out.get(k, 0) + 1
        return out

    @app.get("/api/demos")
    def demos():
        return [describe(HarnessRef(demo=d)) | {"demo": d} for d in list_demos()]

    @app.post("/api/harness/inspect")
    def inspect(ref: HarnessRef):
        try:
            return describe(ref)
        except HarnessError as e:
            raise HTTPException(400, str(e)) from e

    # ---- experiments --------------------------------------------------------------------------------------
    @app.post("/api/experiments")
    def create(cfg: ExperimentConfig):
        rec = mgr.create(cfg)
        return rec.meta()

    @app.get("/api/experiments")
    def list_experiments():
        return [r.meta() for r in sorted(mgr.records.values(), key=lambda r: -r.created)]

    def need(rid: str):
        rec = mgr.get(rid)
        if rec is None:
            raise HTTPException(404, "unknown experiment")
        return rec

    @app.get("/api/experiments/{rid}")
    def get(rid: str):
        rec = need(rid)
        return {**rec.meta(), "config": rec.cfg.model_dump(), "progress": rec.progress, "summary": mgr.summary(rec)}

    @app.post("/api/experiments/{rid}/cancel")
    def cancel(rid: str):
        need(rid).cancel.set()
        return {"ok": True}

    @app.delete("/api/experiments/{rid}")
    def delete(rid: str):
        need(rid)
        mgr.delete(rid)
        return {"ok": True}

    @app.get("/api/experiments/{rid}/events")
    async def events(rid: str, after: int = 0):
        rec = need(rid)

        async def gen():
            i = after
            while True:
                while i < len(rec.events):
                    ev = rec.events[i]
                    i += 1
                    yield f"id: {i}\ndata: {json.dumps(ev, default=str)}\n\n"
                if rec.status not in ("queued", "running") and i >= len(rec.events):
                    yield f"data: {json.dumps({'type': 'closed', 'status': rec.status})}\n\n"
                    return
                await asyncio.sleep(0.4)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/experiments/{rid}/tasks")
    def task_list(rid: str):
        rec = need(rid)
        tasks = mgr.tasks(rec)
        per: dict[str, dict[str, Any]] = {}
        for r in mgr.rows(rec):
            t = per.setdefault(r["task_id"], {"id": r["task_id"], "arms": {}})
            t["arms"][r["arm"]] = {"score": r["score"], "e2e_ms": r["e2e_ms"], "llm_calls": r["llm_calls"],
                                   "error": bool(r["error"])}
        out = []
        for tid, t in per.items():
            task = tasks.get(tid, {})
            t["preview"] = str(task.get("message") or task.get("input") or task.get("question") or json.dumps(task)[:120])[:140]
            t["kind"] = task.get("kind", "")
            out.append(t)
        return out

    @app.get("/api/experiments/{rid}/task/{task_id}")
    def task_detail(rid: str, task_id: str):
        rec = need(rid)
        rows = [r for r in mgr.rows(rec) if r["task_id"] == task_id]
        if not rows:
            raise HTTPException(404, "no rows for that task")
        return {"task": mgr.tasks(rec).get(task_id, {}), "rows": rows}

    @app.get("/api/experiments/{rid}/rows.jsonl")
    def rows_file(rid: str):
        p = need(rid).dir / "rows.jsonl"
        if not p.exists():
            raise HTTPException(404, "no rows yet")
        return FileResponse(p, media_type="application/x-ndjson", filename=f"{rid}-rows.jsonl")

    # ---- imported call logs ---------------------------------------------------------------------------------
    MAX_TRACE_BYTES = 64_000_000
    _cache: dict[str, tuple[float, Any]] = {}

    def trace_path(req: TraceReq) -> Path:
        if req.text is not None:
            if len(req.text.encode()) > MAX_TRACE_BYTES:
                raise HTTPException(413, f"log is larger than {MAX_TRACE_BYTES // 1_000_000} MB; "
                                        "give a path on this machine instead, or export fewer tasks")
            d = state.home() / "traces"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{hashlib.sha1(req.text.encode()).hexdigest()[:12]}-{Path(req.filename).name}"
            if not p.exists():
                p.write_text(req.text)
            return p
        if not req.path:
            raise HTTPException(400, "give a path to the log file, or upload it")
        return Path(req.path).expanduser()

    def parsed(req: TraceReq):
        p = trace_path(req)
        try:
            mtime = p.stat().st_mtime
        except OSError as e:
            raise HTTPException(400, f"cannot read {p}: {e}") from e
        key = f"{p}|{mtime}|{req.limit_tasks}"
        hit = _cache.get(key)
        if hit is None:
            try:
                calls = trace.load_trace(p, req.limit_tasks)
            except trace.TraceError as e:
                raise HTTPException(400, str(e)) from e
            hit = (mtime, (calls, trace.analyze(calls, str(p))))
            _cache.clear()  # only the current log matters; these hold every prompt in the file
            _cache[key] = hit
        return hit[1]

    def projection(req: ProjectReq, report) -> dict[str, Any]:
        if req.avg_prompt_tokens or req.avg_output_tokens:  # the user's own measured averages win
            for s_ in report.sites:
                if req.avg_prompt_tokens:
                    s_.med_prompt_tokens = req.avg_prompt_tokens
                    s_.total_prompt_tokens = req.avg_prompt_tokens * s_.n
                if req.avg_output_tokens:
                    s_.med_out_tokens = req.avg_output_tokens
                    s_.total_out_tokens = req.avg_output_tokens * s_.n
        accept = {k: v for k, v in req.accept.items() if v != "generation"}
        return trace.project(report, accept, req.llm_price_in, req.llm_price_out,
                             req.decider_price_in, req.decider_price_out)

    @app.get("/api/trace/examples")
    def trace_examples():
        """Call logs bundled with the repo, so the Import page can be tried without exporting anything."""
        root = Path(__file__).resolve().parent.parent.parent / "examples" / "traces"
        return [{"path": str(p), "name": p.name, "size_kb": round(p.stat().st_size / 1024)}
                for p in sorted(root.glob("*.jsonl"))] if root.is_dir() else []

    @app.post("/api/trace/inspect")
    def trace_inspect(req: TraceReq):
        _, report = parsed(req)
        return {"report": trace.report_json(report), "path": report.path,
                "suggested": {s.site: s.kind for s in report.movable()}}

    @app.post("/api/trace/project")
    def trace_project(req: ProjectReq):
        _, report = parsed(req)
        return projection(req, report)

    @app.post("/api/trace/build")
    def trace_build(req: BuildReq):
        calls, report = parsed(req)
        accept = {k: v for k, v in req.accept.items() if v != "generation"}
        if not accept:
            raise HTTPException(400, "choose at least one step to move to the decision model")
        out_dir = state.home() / "imported" / slug(req.name, "imported")
        try:
            built = imported.build(report, calls, accept, out_dir, req.name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {**built, "projection": projection(req, report), "name": req.name}

    # ---- models -------------------------------------------------------------------------------------------
    @app.get("/api/models")
    def models():
        return {"local": modelhub.scan_local(), "catalog": modelhub.CATALOG, "servers": modelhub.list_servers(),
                "jobs": modelhub.jobs_json()}


    @app.post("/api/models/pull")
    def pull(req: PullReq):
        try:
            return modelhub.start_pull(req.repo_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e


    @app.post("/api/models/serve")
    def serve(req: ServeReq):
        try:
            return modelhub.serve(req.model, port=req.port, gpu_util=req.gpu_util, max_len=req.max_len,
                                  max_seqs=req.max_seqs, served_name=req.served_name)
        except RuntimeError as e:
            raise HTTPException(400, str(e)) from e


    @app.post("/api/models/stop")
    def stop(req: StopReq):
        try:
            modelhub.stop(req.name)
        except RuntimeError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

    @app.get("/api/models/logs/{name}")
    def logs(name: str):
        return {"logs": modelhub.logs(name)}

    # ---- frontend -----------------------------------------------------------------------------------------
    if (WEBUI / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=WEBUI / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (WEBUI / path).resolve()
            if path and f.is_file() and WEBUI in f.parents:
                return FileResponse(f)
            return FileResponse(WEBUI / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_ui():
            return JSONResponse({"detail": "frontend not built - run: cd frontend && npm install && npm run build"})

    return app
