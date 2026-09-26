"""Local model hub: find models on disk, pull them from Hugging Face, serve them with vLLM in Docker.

Everything here is a convenience around "an OpenAI-compatible endpoint" - if you already run vLLM, Ollama or
llama.cpp yourself, skip this page and paste the URL in Setup.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.types import slug
from . import state

IMAGE = os.environ.get("JEVCONTROL_VLLM_IMAGE", "vllm/vllm-openai:nightly-aarch64")
PREFIX = os.environ.get("JEVCONTROL_CONTAINER_PREFIX", "jevcontrol-")
LABEL = "jevcontrol=1"

# Suggestions shown in the UI. `role` only steers the default settings; any model can play either role.
CATALOG = [
    {"repo_id": "abhishek085/spark-s1-4b-v6-nvfp4", "role": "decision", "title": "spark-s1 4B (NVFP4)",
     "note": "System One decision model: trained for single-token menu answers. ~5 GB."},
    {"repo_id": "google/gemma-4-E4B-it", "role": "llm", "title": "Gemma 4 E4B (instruct)",
     "note": "Small general model: works as the main LLM, or zero-shot as a decision model. ~15 GB."},
    {"repo_id": "Qwen/Qwen3-4B", "role": "llm", "title": "Qwen3 4B",
     "note": "Small general model; can also be tried as a zero-shot decision model."},
]


# ---------------------------------------------------------------------------------------------------- local
def _dir_size(p: Path) -> int:
    total = 0
    try:
        for root, _, files in os.walk(p, followlinks=True):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _describe(path: Path, mid: str, source: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    quant = None
    try:
        quant = json.loads((path / "hf_quant_config.json").read_text())["quantization"]["quant_algo"]
    except (OSError, json.JSONDecodeError, KeyError):
        quant = (cfg.get("quantization_config") or {}).get("quant_method")
    return {"id": mid, "path": str(path), "source": source, "size_gb": round(_dir_size(path) / 1e9, 1),
            "arch": (cfg.get("architectures") or ["?"])[0], "quant": quant,
            "decision_model": "spark-s1" in mid.lower()}


def hf_cache_dirs() -> list[Path]:
    dirs = []
    for c in (os.environ.get("HF_HUB_CACHE"), os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
              str(Path.home() / ".cache/huggingface/hub"), str(state.home() / "hf/hub")):
        if c and Path(c).is_dir() and Path(c) not in dirs:
            dirs.append(Path(c))
    return dirs


_SCAN_CACHE: dict[str, Any] = {"key": None, "t": 0.0, "val": []}


def scan_local(max_age: float = 20.0) -> list[dict[str, Any]]:
    """Cached for a few seconds: the UI polls this and sizing a few hundred GB of snapshots is not free."""
    key = (str(state.models_dir()), tuple(map(str, hf_cache_dirs())))
    if _SCAN_CACHE["key"] == key and time.time() - _SCAN_CACHE["t"] < max_age:
        return _SCAN_CACHE["val"]
    val = _scan_local()
    _SCAN_CACHE.update(key=key, t=time.time(), val=val)
    return val


def _scan_local() -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    md = state.models_dir()
    if md.is_dir():
        for d in sorted(md.iterdir()):
            if (d / "config.json").exists():
                out[d.name] = _describe(d, d.name, "models/")
    for cache in hf_cache_dirs():
        for d in sorted(cache.glob("models--*")):
            snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
            snap = next((s for s in snaps if (s / "config.json").exists()), None)
            if snap is None:
                continue
            rid = d.name[len("models--"):].replace("--", "/", 1)
            if rid not in out and Path(rid).name not in out:
                out[rid] = _describe(snap, rid, "HF cache")
    return sorted(out.values(), key=lambda m: m["id"].lower())


def resolve_local(model: str) -> dict[str, Any] | None:
    return next((m for m in scan_local() if m["id"] == model or Path(m["id"]).name == Path(model).name), None)


# ---------------------------------------------------------------------------------------------------- pull
@dataclass
class PullJob:
    id: str
    repo_id: str
    status: str = "running"  # running | done | error
    bytes_done: int = 0
    bytes_total: int = 0
    error: str = ""
    dest: str = ""
    started: float = field(default_factory=time.time)


JOBS: dict[str, PullJob] = {}


def start_pull(repo_id: str) -> PullJob:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise ValueError("repo id looks like 'org/name'")
    job = PullJob(uuid.uuid4().hex[:8], repo_id, dest=str(state.models_dir() / repo_id.split("/")[1]))
    JOBS[job.id] = job

    def work() -> None:
        try:
            # ~/.cache/huggingface is root-owned on some boxes (vLLM containers write it), so keep our own cache.
            os.environ.setdefault("HF_HOME", str(state.home() / "hf"))
            from huggingface_hub import HfApi, snapshot_download

            token = os.environ.get("HF_TOKEN") or None
            try:
                info = HfApi().model_info(repo_id, files_metadata=True, token=token)
                job.bytes_total = sum((s.size or 0) for s in info.siblings)
            except Exception:  # noqa: BLE001
                pass
            stop = threading.Event()

            def watch() -> None:
                while not stop.wait(1.0):
                    job.bytes_done = _dir_size(Path(job.dest))

            threading.Thread(target=watch, daemon=True).start()
            try:
                snapshot_download(repo_id, local_dir=job.dest, token=token)
            finally:
                stop.set()
            job.bytes_done = _dir_size(Path(job.dest))
            _SCAN_CACHE["key"] = None  # a new model is on disk: rescan on the next request
            job.status = "done"
        except Exception as e:  # noqa: BLE001
            job.status, job.error = "error", f"{type(e).__name__}: {e}"

    threading.Thread(target=work, daemon=True).start()
    return job


# ---------------------------------------------------------------------------------------------------- serve
def docker_ok() -> tuple[bool, str]:
    """Can this machine serve a model from the app? Only Linux with an NVIDIA GPU can.

    Docker cannot pass a GPU through on macOS or Windows, and the vLLM image is CUDA-only, so serving from the
    app is a Linux+NVIDIA feature. Everywhere else you run the server yourself (llama.cpp, LM Studio, Ollama,
    MLX, a hosted API) and paste its URL - every other part of JevControl works the same.
    """
    if platform.system() != "Linux":
        return False, (f"serving from the app needs Linux with an NVIDIA GPU (this is {platform.system()}). "
                       "Run your own OpenAI-compatible server - llama.cpp, LM Studio, Ollama, MLX - and paste "
                       "its URL in Setup; JevControl will find it on a common port.")
    if not shutil.which("docker"):
        return False, "docker is not installed - run your own vLLM/Ollama and paste its URL in Setup"
    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    return (r.returncode == 0, "" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["docker not usable"])[-1])


def _free_port(start: int = 8101) -> int:
    for p in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def _mem_gb() -> tuple[float, float]:
    """(available, total) GB. Available is 0.0 when this platform will not say, and callers must not read that
    as "no memory" - /proc/meminfo is Linux-only."""
    try:
        vals = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":")
            vals[k] = int(v.split()[0]) / 1024 / 1024
        return vals.get("MemAvailable", 0.0), vals.get("MemTotal", 0.0)
    except OSError:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except (OSError, ValueError):
            total = 0.0
        return 0.0, total


def _health(port: int) -> list[str] | None:
    """Model ids the server reports, or None if it is not answering (yet)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])] if r.status == 200 else None
    except Exception:  # noqa: BLE001
        return None


COMMON_PORTS = (8000, 8001, 8002, 8004, 8010, 8080, 8101, 8102, 8103, 8355, 8400, 11434, 1234)


def detect_endpoints(skip_ports: set[int]) -> list[dict[str, Any]]:
    """OpenAI-compatible servers already answering on common local ports (vLLM, Ollama, LM Studio, SGLang...)."""
    out: list[dict[str, Any]] = []
    for port in COMMON_PORTS:
        if port in skip_ports:
            continue
        ids = _health(port)
        if ids:
            out.append({"name": f"port-{port}", "status": "running", "running": True, "port": port, "model": ids[0],
                        "served_name": ids[0], "ready": True, "base_url": f"http://localhost:{port}/v1", "managed": False})
    return out


def list_servers() -> list[dict[str, Any]]:
    managed = _docker_servers()
    return managed + detect_endpoints({s["port"] for s in managed})


def _docker_servers() -> list[dict[str, Any]]:
    ok, _ = docker_ok()
    if not ok:
        return []
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"label={LABEL}", "--format", "{{json .}}"],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        c = json.loads(line)
        labels = dict(kv.split("=", 1) for kv in c.get("Labels", "").split(",") if "=" in kv)
        m = re.search(r":(\d+)->8000", c.get("Ports", ""))
        port = int(m.group(1)) if m else int(labels.get("jevcontrol.port", 0) or 0)
        running = c.get("State") == "running"
        ids = _health(port) if running and port else None
        out.append({"name": c["Names"], "status": c.get("Status", ""), "running": running, "port": port,
                    "model": labels.get("jevcontrol.model", "") or (ids[0] if ids else ""),
                    "served_name": labels.get("jevcontrol.served", "") or (ids[0] if ids else c["Names"]),
                    "ready": ids is not None, "managed": True,
                    "base_url": f"http://localhost:{port}/v1" if port else ""})
    return out


def serve(model: str, *, port: int | None = None, gpu_util: float = 0.15, max_len: int = 8192,
          max_seqs: int = 16, served_name: str | None = None, extra_args: list[str] | None = None) -> dict[str, Any]:
    ok, why = docker_ok()
    if not ok:
        raise RuntimeError(why)
    local = resolve_local(model)
    if local is None:
        raise RuntimeError(f"{model!r} is not on this machine yet - pull it first")
    served = served_name or slug(Path(local["id"]).name)
    name = PREFIX + slug(served)
    if any(s["name"] == name for s in _docker_servers()):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    avail, total = _mem_gb()
    need = gpu_util * total + 4
    if avail and avail < need:  # avail == 0 means this platform did not report it, not that memory is full
        raise RuntimeError(f"not enough free memory: need ~{need:.0f} GB (gpu_util {gpu_util} x {total:.0f} GB + 4), "
                           f"{avail:.0f} GB available. Stop another model or lower gpu_util.")
    port = port or _free_port()
    cmd = ["docker", "run", "-d", "--name", name, "--label", LABEL, "--label", f"jevcontrol.model={local['id']}",
           "--label", f"jevcontrol.served={served}", "--gpus", "all", "--ipc", "host", "-p", f"{port}:8000",
           "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "-e", "HF_HUB_OFFLINE=1"]
    path = Path(local["path"])
    if local["source"] == "models/":
        cmd += ["-v", f"{path}:/models/{path.name}:ro"]
        model_arg = f"/models/{path.name}"
    else:  # HF cache: mount it whole and refer to the repo id
        cache = next(c for c in hf_cache_dirs() if str(path).startswith(str(c)))
        cmd += ["-v", f"{cache.parent}:/root/.cache/huggingface"]
        model_arg = local["id"]
    for extra in ("vllm", "flashinfer"):
        d = Path.home() / ".cache" / extra
        if d.is_dir():
            cmd += ["-v", f"{d}:/root/.cache/{extra}"]
    cmd += [IMAGE, "--model", model_arg, "--served-model-name", served, "--host", "0.0.0.0", "--port", "8000",
            "--gpu-memory-utilization", str(gpu_util), "--max-model-len", str(max_len), "--max-num-seqs", str(max_seqs),
            "--trust-remote-code", *(extra_args or [])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "docker run failed")
    return {"name": name, "port": port, "served_name": served, "base_url": f"http://localhost:{port}/v1"}


def stop(name: str) -> None:
    if not any(s["name"] == name for s in _docker_servers()):
        raise RuntimeError(f"{name} is not a JevControl-managed server")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def logs(name: str, tail: int = 60) -> str:
    if not any(s["name"] == name for s in _docker_servers()):
        return ""
    r = subprocess.run(["docker", "logs", "--tail", str(tail), name], capture_output=True, text=True)
    return re.sub(r"\x1b\[[0-9;]*m", "", (r.stdout + r.stderr))[-6000:]


def jobs_json() -> list[dict[str, Any]]:
    return [asdict(j) for j in sorted(JOBS.values(), key=lambda j: -j.started)]
