#!/usr/bin/env bash
# One-command demo on Apple Silicon: convert spark-s1 to MLX (first run only), serve it with mlx_lm.server,
# and start JevControl against it and an Ollama model you already have. Everything is local.
#
#   scripts/run_demo_mac.sh                          # start spark-s1 + app (main LLM: $LLM_MODEL via Ollama)
#   scripts/run_demo_mac.sh stop                      # stop the spark-s1 server started by this script
#
# Env: LLM_MODEL (an Ollama model you have pulled, e.g. "gemma4:12b-mlx" - no default, see below)
#      DEC_REPO (default abhishek085/spark-s1-4b-v6), DEC_DIR (default models/<DEC_REPO's name>)
#
# Requires: an MLX venv with mlx-lm >= 0.31 at .venv-mlx (`python -m venv .venv-mlx && .venv-mlx/bin/pip
# install mlx-lm`), Ollama running on :11434 with LLM_MODEL already pulled, and this repo's own .venv
# (`pip install -e ".[dev]"`) for `jevcontrol serve`. See docs/MODELS.md for why each of these is needed
# on a Mac - the short version is: no Docker/CUDA path here, so both models are plain HTTP servers you run
# yourself and JevControl just points at.
set -euo pipefail
cd "$(dirname "$0")/.."

DEC_REPO="${DEC_REPO:-abhishek085/spark-s1-4b-v6}"
DEC_DIR="${DEC_DIR:-models/$(basename "$DEC_REPO")}"
MLX_DIR="${DEC_DIR}-mlx-8bit"
PIDFILE=".jevcontrol/run_demo_mac.spark-s1.pid"

if [ "${1:-}" = "stop" ]; then
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"
  echo "stopped spark-s1 (the app and Ollama are left running)"; exit 0
fi

[ -x .venv-mlx/bin/mlx_lm.server ] || { echo "missing .venv-mlx - see the header of this script"; exit 1; }
[ -x .venv/bin/jevcontrol ] && . .venv/bin/activate

if [ -z "${LLM_MODEL:-}" ]; then
  echo "set LLM_MODEL to an Ollama model you have pulled, e.g.:"
  echo "  LLM_MODEL=gemma4:12b-mlx scripts/run_demo_mac.sh"
  echo "models Ollama already has:"
  ollama list 2>/dev/null | tail -n +2 | awk '{print "  " $1}' || echo "  (could not reach Ollama on :11434 - is it running?)"
  exit 1
fi
curl -sf -m 2 http://localhost:11434/v1/models >/dev/null || { echo "Ollama is not answering on :11434 - start it first"; exit 1; }

if [ ! -d "$MLX_DIR" ]; then
  echo "converting $DEC_REPO to 8-bit MLX (first run only; ~4-5 min, ~4.2 GB on disk) ..."
  [ -d "$DEC_DIR" ] || .venv-mlx/bin/hf download "$DEC_REPO" --local-dir "$DEC_DIR"
  # v6 ships as model_type "qwen3_5_text", which mlx-lm's loader does not recognise; this only renames the
  # key for the local copy's config.json so MLX finds its Qwen3.5 implementation. Weights are untouched.
  python3 -c "
import json, pathlib
p = pathlib.Path('$DEC_DIR/config.json')
c = json.load(p.open())
c['model_type'] = 'qwen3_5'
json.dump(c, p.open('w'), indent=2)
"
  .venv-mlx/bin/mlx_lm.convert --hf-path "$DEC_DIR" --mlx-path "$MLX_DIR" -q --q-bits 8
fi

if curl -sf -m 2 http://localhost:8102/v1/models >/dev/null 2>&1; then
  echo "spark-s1 already serving on :8102"
else
  echo "starting spark-s1 on :8102 ..."
  mkdir -p .jevcontrol
  nohup .venv-mlx/bin/mlx_lm.server --model "$MLX_DIR" --port 8102 >.jevcontrol/spark-s1.log 2>&1 &
  echo $! > "$PIDFILE"
  until curl -sf -m 2 http://localhost:8102/v1/models >/dev/null 2>&1; do sleep 1; done
fi

echo "main LLM: $LLM_MODEL (Ollama, :11434) · decision model: spark-s1-4b-v6 (MLX, :8102)"
echo "starting JevControl on http://localhost:8600"
echo "In Setup: Auto-fill from running servers, then Test connection on both - the Ollama one may need"
echo "'Turn thinking mode off' ticked if $LLM_MODEL reasons before answering (the connection test says so)."
exec jevcontrol serve --port 8600
