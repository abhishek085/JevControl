"""A deterministic OpenAI-compatible stub so the whole stack can be tested without a GPU.

Behaviour is driven by markers in the prompt, e.g. a state containing ``PICK:orders`` makes both the JSON
(prompted) path and the logprob (menu) path choose the option labelled ``orders``.
"""

from __future__ import annotations

import json
import math
import re
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request


def make_app(menu_conf: float = 0.9, name: str = "stub", force: str | None = None) -> FastAPI:
    app = FastAPI()
    counters = {"chat": 0, "logprobs": 0}
    app.state.counters = counters

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": name}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        text = "\n".join(m["content"] for m in body["messages"])
        pick = re.search(r"PICK:([\w-]+)", text)
        want = force or (pick.group(1) if pick else None)
        if body.get("logprobs"):
            counters["logprobs"] += 1
            opts = re.findall(r"^([A-Z])\. (.+)$", text, re.MULTILINE)
            letters = [o[0] for o in opts]
            labels = [o[1] for o in opts]
            idx = 0
            for i, lab in enumerate(labels):
                if want and (want == lab or (want == "true" and lab.startswith("Yes")) or (want == "false" and lab.startswith("No"))):
                    idx = i
            rest = (1 - menu_conf) / max(len(letters) - 1, 1)
            top = [{"token": L, "logprob": math.log(menu_conf if i == idx else rest)} for i, L in enumerate(letters)] or \
                  [{"token": "A", "logprob": -0.1}]
            return {"choices": [{"message": {"content": letters[idx] if letters else "A"},
                                 "logprobs": {"content": [{"token": top[idx]["token"], "logprob": top[idx]["logprob"], "top_logprobs": top}]}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 1}}
        counters["chat"] += 1
        time.sleep(0.01)
        if "Answer with the single letter" in text:  # the menu question, asked of a chat-only endpoint
            opts = re.findall(r"^([A-Z])\. (.+)$", text, re.MULTILINE)
            idx = 0
            for i, (_letter, lab) in enumerate(opts):
                if want and (want == lab or (want == "true" and lab.startswith("Yes")) or (want == "false" and lab.startswith("No"))):
                    idx = i
            letter = opts[idx][0] if opts else "A"
            return {"choices": [{"message": {"content": letter}}], "usage": {"prompt_tokens": 120, "completion_tokens": 1}}
        if "Reply with JSON" in text:
            if "true|false" in text:
                ans = "true" if want == "true" else "false"
                return {"choices": [{"message": {"content": json.dumps({"answer": ans})}}], "usage": {"prompt_tokens": 150, "completion_tokens": 8}}
            m = re.search(r"Options:\n((?:- .*\n?)+)", text)
            labels = [re.match(r"- ([^:\n]+)", l).group(1).strip() for l in (m.group(1).strip().splitlines() if m else [])]
            ans = want if want in labels else (labels[0] if labels else "x")
            return {"choices": [{"message": {"content": json.dumps({"answer": ans})}}], "usage": {"prompt_tokens": 150, "completion_tokens": 8}}
        return {"choices": [{"message": {"content": "You have 30 days to return items."}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 12}}

    return app


class StubServer:
    def __init__(self, **kw):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.app = make_app(**kw)
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("stub did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def make_jev_app(name: str = "jev-stub", confidence: float = 0.88, path: str = "/v1/decide") -> FastAPI:
    """A typed decision API in the shape of an open-spark-Jev gateway, for testing the `jev` endpoint kind.

    `path` picks which of the two shapes this server speaks, so both code paths can be exercised.
    """
    app = FastAPI()
    counters = {"decide": 0, "evaluate": 0}
    app.state.counters = counters

    def pick(labels: list[str], text: str) -> int:
        want = re.search(r"PICK:([\w-]+)", text)
        if want:
            w = want.group(1)
            for i, lab in enumerate(labels):
                if lab.lower() == w.lower():
                    return i
        return 0

    def spread(labels: list[str], idx: int) -> dict[str, float]:
        rest = (1 - confidence) / max(len(labels) - 1, 1)
        return {lab: (confidence if i == idx else rest) for i, lab in enumerate(labels)}

    if path == "/v1/decide":
        @app.post("/v1/decide")
        async def decide(req: Request):
            body = await req.json()
            counters["decide"] += 1
            q = body["questions"][0]
            state = json.dumps(body["state"])
            if q["type"] == "boolean":
                labels = ["true", "false"]
            elif q["type"] == "score":
                labels = [str(x["value"]) for x in q["levels"]]
            else:
                labels = [o["id"] for o in q["options"]]
            idx = pick(labels, state)
            probs = spread(labels, idx)
            return {"model": name, "latency_ms": 12.0,
                    "decisions": {q["id"]: {"selected": labels[idx], "probabilities": probs,
                                            "confidence": probs[labels[idx]], "latency_ms": 12.0}}}
    else:
        @app.post("/v1/evaluate")
        async def evaluate(req: Request):
            body = await req.json()
            counters["evaluate"] += 1
            name_, q = next(iter(body["questions"].items()))
            state = json.dumps(body["state"])
            if q["type"] in ("noul", "boolean"):
                p = confidence if pick(["true", "false"], state) == 0 else 1 - confidence
                return {"model": name, "answers": {name_: {"type": "noul", "noul": p, "probability": p}}}
            labels = list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(i) for i in range(len(q["criteria"]))]
            idx = pick(labels, state)
            probs = spread(labels, idx)
            key = "choice" if q["type"] == "choice" else "score"
            return {"model": name, "answers": {name_: {"type": q["type"], key: labels[idx],
                                                       "probabilities": probs, "confidence": probs[labels[idx]]}}}

    return app


class JevStub(StubServer):
    def __init__(self, **kw):
        path = kw.pop("path", "/v1/decide")
        super().__init__()
        self.app = make_jev_app(path=path, **kw)
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
