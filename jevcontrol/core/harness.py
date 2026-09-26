"""Loading a harness and the ``ctx`` object it runs against.

A harness is one Python file:

    META  = {"name": "...", "description": "...", "sites": {"route": "which tool answers this"}}   # optional
    TOOLS = {"search": search_fn}                                                                  # optional
    def run(task: dict, ctx) -> Any: ...            # required: your agent, decisions via ctx.decide.*
    def score(task: dict, output) -> bool | float   # optional: else the built-in scorer compares to task["expected"]

and a ``tasks.jsonl`` next to it (one JSON object per line; an optional ``truth`` key holds ground truth per
decision site, which unlocks decision-level accuracy in the report).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decide import BoundLLM, Decide
from .recorder import Recorder, ToolCall
from .toolcache import ToolCache
from .types import HarnessRef

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


class HarnessError(RuntimeError):
    pass


@dataclass
class Harness:
    id: str
    path: Path
    tasks_path: Path
    run: Callable[[dict[str, Any], Any], Any]
    score: Callable[[dict[str, Any], Any], Any] | None
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.id

    def load_tasks(self, limit: int | None = None) -> list[dict[str, Any]]:
        return load_tasks(self.tasks_path, limit)


def load_tasks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise HarnessError(f"tasks file not found: {path}")
    tasks = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError as e:
            raise HarnessError(f"{path}:{i + 1}: invalid JSON ({e})") from e
        if not isinstance(t, dict):
            raise HarnessError(f"{path}:{i + 1}: each task must be a JSON object")
        t.setdefault("id", f"t{len(tasks) + 1:04d}")
        tasks.append(t)
    if not tasks:
        raise HarnessError(f"{path}: no tasks")
    return tasks[:limit] if limit else tasks


def list_demos() -> list[str]:
    return sorted(p.parent.name for p in DEMO_DIR.glob("*/harness.py"))


def load_harness(ref: HarnessRef) -> Harness:
    if ref.demo:
        path = DEMO_DIR / ref.demo / "harness.py"
        hid = ref.demo
    elif ref.path:
        path = Path(ref.path).expanduser().resolve()
        hid = path.parent.name
    else:
        raise HarnessError("harness needs `demo` or `path`")
    if not path.exists():
        raise HarnessError(f"harness file not found: {path}")
    mod_name = f"jevcontrol_harness_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise HarnessError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # let a harness import helper modules next to it
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise HarnessError(f"importing {path} failed: {type(e).__name__}: {e}") from e
    finally:
        sys.path.remove(str(path.parent))
    if not callable(getattr(mod, "run", None)):
        raise HarnessError(f"{path} must define run(task, ctx)")
    tasks_path = Path(ref.tasks).expanduser().resolve() if ref.tasks else path.parent / "tasks.jsonl"
    return Harness(hid, path, tasks_path, mod.run, getattr(mod, "score", None), dict(getattr(mod, "TOOLS", {}) or {}),
                   dict(getattr(mod, "META", {}) or {}))


class Ctx:
    """What ``run(task, ctx)`` receives. The same harness code runs unchanged under every arm."""

    def __init__(self, task: dict[str, Any], llm: BoundLLM, decide: Decide, rec: Recorder,
                 harness: Harness, cache: ToolCache):
        self.task, self.llm, self.decide, self.rec = task, llm, decide, rec
        self._harness, self._cache = harness, cache

    def tool(self, name: str, **args: Any) -> Any:
        fn = self._harness.tools.get(name)
        if fn is None:
            raise HarnessError(f"unknown tool {name!r}; the harness defines: {sorted(self._harness.tools)}")
        out, cached, ms = self._cache.call(name, fn, args)
        self.rec.add_tool(ToolCall(name, args, cached, ms))
        return out
