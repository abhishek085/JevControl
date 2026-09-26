"""Deterministic tool replay.

If the harness hits live tools (search, DB, web), two arms would see different data for reasons that have
nothing to do with the decision model. So tool calls are cached by (tool, args): the first arm to make a call
pays for it; every other arm replays the same result. A different decision that produces *new* arguments is a
real cache miss and runs live - so paths that agree are compared fairly and paths that diverge stay honest.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _key(name: str, args: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps([name, args], sort_keys=True, default=str).encode()).hexdigest()


class ToolCache:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.mem: dict[str, tuple[Any, float]] = {}
        self.lock = threading.Lock()
        if path and path.exists():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.mem[r["k"]] = (r["v"], r["ms"])
                except (json.JSONDecodeError, KeyError):
                    continue

    def call(self, name: str, fn: Callable[..., Any], args: dict[str, Any]) -> tuple[Any, bool, float]:
        k = _key(name, args)
        with self.lock:
            hit = self.mem.get(k)
        if hit is not None:
            return hit[0], True, hit[1]
        t0 = time.perf_counter()
        out = fn(**args)
        ms = (time.perf_counter() - t0) * 1000
        with self.lock:
            self.mem[k] = (out, ms)
            if self.path:
                try:
                    with self.path.open("a") as f:
                        f.write(json.dumps({"k": k, "name": name, "args": args, "v": out, "ms": ms}, default=str) + "\n")
                except TypeError:
                    pass  # unserialisable result: keep it in memory only
        return out, False, ms
