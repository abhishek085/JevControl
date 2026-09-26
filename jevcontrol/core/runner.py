"""Run an experiment: the same harness over the same tasks once per arm, saving every row.

Arms run one after another (so a local GPU is not shared between arms and latencies stay comparable);
tasks inside an arm run ``concurrency`` at a time (1 by default = clean latency).
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .analysis import analyze
from .decide import BoundLLM, Decide, Decider, EscalatingDecider, LLMDecider, MenuDecider
from .harness import Ctx, Harness, load_harness
from .llm import LLMClient
from .recorder import Recorder
from .toolcache import ToolCache
from .types import Arm, ExperimentConfig

Emit = Callable[[dict[str, Any]], None]


def normalize_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Guarantee exactly one baseline arm, first, and unique arm ids."""
    arms = [a for a in cfg.arms if a.kind != "baseline"]
    base = next((a for a in cfg.arms if a.kind == "baseline"), Arm(id="baseline", label="Baseline (LLM decides)", kind="baseline"))
    ordered = [base] + arms
    seen: set[str] = set()
    for a in ordered:
        while a.id in seen:
            a.id += "-2"
        seen.add(a.id)
        if a.kind != "baseline" and a.decider is None:
            raise ValueError(f"arm {a.id!r} ({a.kind}) needs a decider endpoint")
    cfg.arms = ordered
    return cfg


def score_row(cfg: ExperimentConfig, harness: Harness, task: dict[str, Any], output: Any) -> float:
    if cfg.scorer == "harness" and harness.score is not None:
        s = harness.score(task, output)
        if isinstance(s, dict):
            s = s.get("score", 0.0)
        return float(bool(s)) if isinstance(s, bool) else float(s)
    exp = task.get(cfg.expected_field)
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    if exp is None:
        return 0.0
    if cfg.scorer == "exact":
        return float(str(output).strip().lower() == str(exp).strip().lower())
    return float(str(exp).strip().lower() in text.lower())  # "contains" (also the fallback)


def _row(cfg: ExperimentConfig, arm: Arm, task: dict[str, Any], output: Any, err: str | None, wall_ms: float,
         rec: Recorder, score: float) -> dict[str, Any]:
    main = [c for c in rec.calls if c.on_main_llm]
    dec = [c for c in rec.calls if not c.on_main_llm]
    saved = sum(t.latency_ms for t in rec.tools if t.cached)  # cache hits cost ~0 wall, but the tool really took this long
    p_in, p_out = cfg.llm.price_in_per_m, cfg.llm.price_out_per_m
    d_in = arm.decider.price_in_per_m if arm.decider else 0.0
    d_out = arm.decider.price_out_per_m if arm.decider else 0.0
    m_in, m_out = sum(c.prompt_tokens for c in main), sum(c.completion_tokens for c in main)
    x_in, x_out = sum(c.prompt_tokens for c in dec), sum(c.completion_tokens for c in dec)
    return {
        "arm": arm.id, "task_id": task["id"], "score": score, "error": err,
        "output": output if isinstance(output, (str, int, float, bool, dict, list, type(None))) else str(output),
        "wall_ms": wall_ms, "e2e_ms": wall_ms + saved,
        "llm_calls": len(main), "llm_generate_calls": sum(1 for c in main if c.role == "generate"),
        "llm_decide_calls": sum(1 for c in main if c.role == "decide"),
        "llm_prompt_tokens": m_in, "llm_completion_tokens": m_out,
        "decider_calls": len(dec), "decider_ms": sum(c.latency_ms for c in dec),
        "decider_prompt_tokens": x_in,
        "cost_usd": (m_in * p_in + m_out * p_out + x_in * d_in + x_out * d_out) / 1e6,
        **rec.as_dict(),
    }


class Experiment:
    def __init__(self, cfg: ExperimentConfig, run_dir: Path, emit: Emit | None = None,
                 cancel: threading.Event | None = None):
        self.cfg = normalize_config(cfg)
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.emit: Emit = emit or (lambda e: None)
        self.cancel = cancel or threading.Event()
        self.harness = load_harness(cfg.harness)
        self.tasks = self.harness.load_tasks(cfg.n_tasks)
        self.cache = ToolCache(self.dir / "tool_cache.jsonl")
        self.clients: dict[str, LLMClient] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {a.id: [] for a in self.cfg.arms}
        self._rows_lock = threading.Lock()

    def client(self, ep: Any) -> LLMClient:
        k = f"{ep.base_url}|{ep.model}|{json.dumps(ep.extra_body, sort_keys=True)}"
        if k not in self.clients:
            self.clients[k] = LLMClient(ep)
        return self.clients[k]

    def decider_for(self, arm: Arm) -> Decider:
        llm = LLMDecider(self.client(self.cfg.llm), self.cfg.capture_text)
        if arm.kind == "baseline":
            return llm
        menu = MenuDecider(self.client(arm.decider), arm.temperature, arm.temperatures)  # type: ignore[arg-type]
        return menu if arm.kind == "menu" else EscalatingDecider(menu, llm, arm.tau, arm.tau_by_site)

    def warmup(self) -> None:
        """One throwaway call per endpoint so the first task doesn't pay server cold-start."""
        self.client(self.cfg.llm).chat("ok", max_tokens=1)
        for a in self.cfg.arms:
            if a.decider:
                try:
                    self.client(a.decider).first_token_logprobs([{"role": "user", "content": "Reply A or B: A"}], 5)
                except Exception:  # noqa: BLE001 - surfaced properly on the first real task
                    pass

    def run_task(self, arm: Arm, decider: Decider, task: dict[str, Any]) -> dict[str, Any]:
        rec = Recorder()
        llm = BoundLLM(self.client(self.cfg.llm), rec, "generate", None, self.cfg.capture_text)
        ctx = Ctx(task, llm, Decide(decider, rec, task.get("truth")), rec, self.harness, self.cache)
        t0 = time.perf_counter()
        out, err = None, None
        try:
            out = self.harness.run(task, ctx)
        except Exception as e:  # noqa: BLE001 - a crashing task is a failed task, not a crashed experiment
            err = f"{type(e).__name__}: {e}"
            if not isinstance(e, (RuntimeError, KeyError, ValueError)):
                err += "\n" + traceback.format_exc(limit=3)
        wall = (time.perf_counter() - t0) * 1000
        try:
            score = 0.0 if err else score_row(self.cfg, self.harness, task, out)
        except Exception as e:  # noqa: BLE001
            score, err = 0.0, f"scorer failed: {type(e).__name__}: {e}"
        return _row(self.cfg, arm, task, out, err, wall, rec, score)

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        (self.dir / "config.json").write_text(cfg.model_dump_json(indent=2))
        rows_path = self.dir / "rows.jsonl"
        rows_path.write_text("")
        self.emit({"type": "start", "arms": [a.id for a in cfg.arms], "n_tasks": len(self.tasks)})
        self.emit({"type": "phase", "text": "Warming up endpoints"})
        self.warmup()
        t_start = time.perf_counter()
        for arm in cfg.arms:
            if self.cancel.is_set():
                break
            decider = self.decider_for(arm)
            self.emit({"type": "arm_start", "arm": arm.id, "label": arm.title(), "n": len(self.tasks)})
            done = 0

            def work(task: dict[str, Any], arm: Arm = arm, decider: Decider = decider) -> dict[str, Any] | None:
                if self.cancel.is_set():
                    return None
                return self.run_task(arm, decider, task)

            with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
                for row in pool.map(work, self.tasks):
                    if row is None:
                        continue
                    with self._rows_lock:
                        self.rows[arm.id].append(row)
                        with rows_path.open("a") as f:
                            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    done += 1
                    self.emit({"type": "task", "arm": arm.id, "task_id": row["task_id"], "score": row["score"],
                               "error": row["error"], "e2e_ms": row["e2e_ms"], "llm_calls": row["llm_calls"],
                               "done": done, "n": len(self.tasks)})
            result = analyze(cfg, self.rows)
            (self.dir / "summary.json").write_text(json.dumps(result, indent=1, default=str))
            self.emit({"type": "arm_done", "arm": arm.id})
            self.emit({"type": "summary", "summary": result})
        result = analyze(cfg, self.rows)
        result["wall_s"] = time.perf_counter() - t_start
        result["cancelled"] = self.cancel.is_set()
        (self.dir / "summary.json").write_text(json.dumps(result, indent=1, default=str))
        for c in self.clients.values():
            c.close()
        self.emit({"type": "done", "cancelled": self.cancel.is_set()})
        return result
