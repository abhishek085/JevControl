"""Runs experiments in background threads and keeps their state on disk (so a restart loses nothing)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.harness import Harness, HarnessError, load_harness
from ..core.llm import LLMClient
from ..core.runner import Experiment, normalize_config
from ..core.types import ExperimentConfig, slug
from . import state


class PreflightError(ValueError):
    pass


@dataclass
class Record:
    id: str
    dir: Path
    cfg: ExperimentConfig
    status: str = "queued"  # queued | running | done | error | cancelled | interrupted
    created: float = field(default_factory=time.time)
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, dict[str, int]] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)

    def meta(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.cfg.name, "status": self.status, "created": self.created, "error": self.error,
                "arms": [a.title() for a in self.cfg.arms], "n_tasks": self.cfg.n_tasks}


def preflight(cfg: ExperimentConfig) -> Harness:
    """Fail fast, with a message a person can act on, before spending minutes of GPU time."""
    try:
        h = load_harness(cfg.harness)
        h.load_tasks(cfg.n_tasks)
    except HarnessError as e:
        raise PreflightError(str(e)) from e
    if not any(a.kind != "baseline" for a in cfg.arms):
        raise PreflightError("add at least one decision-model arm to compare against the baseline")
    checks = [("main LLM", cfg.llm, False)] + [(f"decision model '{a.title()}'", a.decider, True)
                                              for a in cfg.arms if a.kind != "baseline" and a.decider]
    seen: set[tuple[str, str]] = set()
    for label, ep, need_lp in checks:
        if (ep.base_url, ep.model) in seen and not need_lp:
            continue
        seen.add((ep.base_url, ep.model))
        c = LLMClient(ep)
        try:
            p = c.probe(need_logprobs=need_lp)
        finally:
            c.close()
        if not p.ok:
            raise PreflightError(f"{label}: {p.error}")
    return h


class Manager:
    def __init__(self) -> None:
        self.root = state.home() / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, Record] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for d in sorted(self.root.iterdir()):
            try:
                cfg = ExperimentConfig.model_validate_json((d / "config.json").read_text())
                st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            except (OSError, ValueError):
                continue
            status = st.get("status", "interrupted")
            if status in ("running", "queued"):
                status = "interrupted"
            rec = Record(d.name, d, cfg, status, st.get("created", d.stat().st_mtime), st.get("error", ""))
            self.records[rec.id] = rec

    def _save_status(self, r: Record) -> None:
        (r.dir / "status.json").write_text(json.dumps({"status": r.status, "created": r.created, "error": r.error}))

    def create(self, cfg: ExperimentConfig) -> Record:
        cfg = normalize_config(cfg)
        preflight(cfg)
        rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(cfg.name, 'run')[:20]}-{uuid.uuid4().hex[:4]}"
        d = self.root / rid
        d.mkdir(parents=True)
        (d / "config.json").write_text(cfg.model_dump_json(indent=2))
        rec = Record(rid, d, cfg)
        self.records[rid] = rec
        self._save_status(rec)
        threading.Thread(target=self._run, args=(rec,), daemon=True).start()
        return rec

    def _emit(self, rec: Record, ev: dict[str, Any]) -> None:
        ev = {**ev, "t": time.time()}
        rec.events.append(ev)
        if ev["type"] == "arm_start":
            rec.progress[ev["arm"]] = {"done": 0, "n": ev["n"]}
        elif ev["type"] == "task":
            rec.progress[ev["arm"]] = {"done": ev["done"], "n": ev["n"]}
        if ev["type"] != "summary":  # summaries are big and re-derivable; keep the log light
            with (rec.dir / "events.jsonl").open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")

    def _run(self, rec: Record) -> None:
        rec.status = "running"
        self._save_status(rec)
        try:
            Experiment(rec.cfg, rec.dir, lambda e: self._emit(rec, e), rec.cancel).run()
            rec.status = "cancelled" if rec.cancel.is_set() else "done"
        except Exception as e:  # noqa: BLE001
            rec.status, rec.error = "error", f"{type(e).__name__}: {e}"
            self._emit(rec, {"type": "error", "error": rec.error})
        self._save_status(rec)

    # ---- reads ------------------------------------------------------------------------------------
    def get(self, rid: str) -> Record | None:
        return self.records.get(rid)

    def summary(self, rec: Record) -> dict[str, Any] | None:
        p = rec.dir / "summary.json"
        return json.loads(p.read_text()) if p.exists() else None

    def rows(self, rec: Record) -> list[dict[str, Any]]:
        p = rec.dir / "rows.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def tasks(self, rec: Record) -> dict[str, dict[str, Any]]:
        try:
            return {t["id"]: t for t in load_harness(rec.cfg.harness).load_tasks(rec.cfg.n_tasks)}
        except HarnessError:
            return {}

    def delete(self, rid: str) -> None:
        import shutil

        rec = self.records.pop(rid, None)
        if rec:
            rec.cancel.set()
            shutil.rmtree(rec.dir, ignore_errors=True)
