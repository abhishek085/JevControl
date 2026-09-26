"""jevcontrol - command line entry point.

    jevcontrol serve                start the app (API + web UI) on http://localhost:8600
    jevcontrol probe URL            check that an OpenAI-compatible endpoint works (and returns logprobs)
    jevcontrol run cfg.yaml         run an experiment headless and print the summary
    jevcontrol trace calls.jsonl    analyse an existing harness's call log: which steps are decisions
    jevcontrol export-trace DIR -o f.jsonl   turn a finished run into a call log (needs capture_text)
"""

from __future__ import annotations

import argparse
import json
import sys


def _probe(a: argparse.Namespace) -> int:
    from .core.llm import LLMClient
    from .core.types import Endpoint

    c = LLMClient(Endpoint(base_url=a.url, model=a.model or "", api_key=a.api_key))
    p = c.probe(need_logprobs=True)
    print(json.dumps(p.__dict__, indent=2))
    return 0 if p.ok else 1


def _run(a: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    import yaml

    from .core.runner import Experiment
    from .core.types import ExperimentConfig
    from .server import state

    cfg = ExperimentConfig.model_validate(yaml.safe_load(Path(a.config).read_text()))
    out = Path(a.out) if a.out else state.home() / "experiments" / f"{time.strftime('%Y%m%d-%H%M%S')}-cli"

    def show(e: dict) -> None:
        if e["type"] == "arm_start":
            print(f"\n== {e['label']} ({e['n']} tasks)")
        elif e["type"] == "task":
            print(f"\r  {e['done']}/{e['n']}", end="", flush=True)

    res = Experiment(cfg, out, show).run()
    (out / "status.json").write_text(json.dumps({"status": "cancelled" if res.get("cancelled") else "done",
                                                 "created": time.time() - res.get("wall_s", 0), "error": ""}))
    print(f"\n\nresults + per-row data: {out}   (open it in the app under Results)")
    for arm in res["arms"]:
        p = res["paired"].get(arm["id"], {})
        extra = (f"  Δacc {p['delta_acc']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}]  speedup {p['speedup']:.2f}x  "
                 f"verdict {p['verdict']}") if p else ""
        print(f"{arm['label']:<32} acc {arm['accuracy']:.3f}  p50 {arm['e2e_ms']['p50']:.0f}ms  "
              f"LLM calls/task {arm['llm_calls']:.1f}{extra}")
    return 0


def _export_trace(a: argparse.Namespace) -> int:
    """Write the LLM-only call log of a finished run's baseline arm, in the format `trace import` reads."""
    from pathlib import Path

    run = Path(a.run)
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line.strip()]
    arm = a.arm or json.loads((run / "config.json").read_text())["arms"][0]["id"]
    out, n, missing = Path(a.out), 0, 0
    with out.open("w") as f:
        for r in rows:
            if r["arm"] != arm:
                continue
            for c in r["calls"]:
                if c.get("prompt") is None:
                    missing += 1
                    continue
                f.write(json.dumps({"task_id": r["task_id"], "site": c["site"] or "generate",
                                    "model": c["model"], "messages": c["prompt"], "output": c["output"],
                                    "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"],
                                    "latency_ms": round(c["latency_ms"], 1)}) + "\n")
                n += 1
    if not n:
        print(f"no captured calls in arm {arm!r}: re-run with capture_text: true in the config", file=sys.stderr)
        return 1
    print(f"wrote {n} calls -> {out}" + (f" ({missing} calls had no captured text)" if missing else ""))
    return 0


def _trace(a: argparse.Namespace) -> int:
    from .core import trace as T

    calls = T.load_trace(a.path, a.limit_tasks)
    rep = T.analyze(calls, str(a.path))
    print(f"{rep.n_calls} calls, {rep.n_tasks} tasks, {len(rep.sites)} steps"
          + (" (token counts estimated from text length)" if rep.tokens_estimated else ""))
    accept = {}
    for s in rep.sites:
        tag = s.kind.upper() if s.movable else "stays on LLM"
        print(f"  {s.site:<22} {tag:<14} {s.n:>4} calls  {s.med_out_tokens:>3} out-tok  {s.reason}")
        if s.movable:
            accept[s.site] = s.kind
    pr = T.project(rep, accept, a.price_in, a.price_out)
    print(f"\nif all {len(accept)} movable steps are accepted, per task:"
          f"\n  main-LLM calls   {pr['llm_calls']['before']:.1f} -> {pr['llm_calls']['after']:.1f}"
          f"\n  main-LLM tokens  {pr['llm_prompt_tokens']['before'] + pr['llm_output_tokens']['before']:.0f}"
          f" -> {pr['llm_prompt_tokens']['after'] + pr['llm_output_tokens']['after']:.0f}"
          f"  ({pr['llm_token_reduction'] * 100:.0f}% less)")
    if pr["priced"]:
        print(f"  cost / 1k tasks  ${pr['cost_per_1k']['before']:.2f} -> ${pr['cost_per_1k']['after']:.2f}")
    print("  latency: run the experiment - it cannot be projected from a log")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevcontrol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8600)
    p = sub.add_parser("probe", help="check an endpoint")
    p.add_argument("url")
    p.add_argument("--model")
    p.add_argument("--api-key", default="EMPTY")
    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("config")
    r.add_argument("--out")
    t = sub.add_parser("trace", help="analyse an existing harness's call log")
    t.add_argument("path")
    t.add_argument("--limit-tasks", type=int)
    t.add_argument("--price-in", type=float, default=0.0, help="$ per M input tokens of your current model")
    t.add_argument("--price-out", type=float, default=0.0)
    e = sub.add_parser("export-trace", help="export a finished run's baseline arm as a call log")
    e.add_argument("run")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--arm")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        import uvicorn

        from .server.api import create_app

        print(f"JevControl on http://{a.host}:{a.port}")
        uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
        return 0
    return {"probe": _probe, "run": _run, "trace": _trace, "export-trace": _export_trace}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
