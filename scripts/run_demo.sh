#!/usr/bin/env bash
# One-command demo on a DGX Spark-class box: serve a small main LLM + the spark-s1 decision model with vLLM (Docker),
# then start JevControl. Everything is local. Re-running is safe (existing containers are reused).
#
#   scripts/run_demo.sh            # start models + app
#   scripts/run_demo.sh stop       # stop the two model containers
#
# Env: LLM_REPO (default google/gemma-4-E4B-it), DEC_DIR (default models/spark-s1-4b-v6-nvfp4)
set -euo pipefail
cd "$(dirname "$0")/.."
LLM_REPO="${LLM_REPO:-google/gemma-4-E4B-it}"
DEC_DIR="${DEC_DIR:-models/spark-s1-4b-v6-nvfp4}"
# The container-name prefix lets a thermal guard that watches `osj-teacher-*` pause them when the box gets hot.
export JEVCONTROL_CONTAINER_PREFIX="${JEVCONTROL_CONTAINER_PREFIX:-osj-teacher-jc-}"
PREFIX="$JEVCONTROL_CONTAINER_PREFIX"

if [ "${1:-}" = "stop" ]; then docker rm -f "${PREFIX}gemma-4-e4b" "${PREFIX}spark-s1" 2>/dev/null || true; exit 0; fi
[ -d "$DEC_DIR" ] || { echo "missing $DEC_DIR - pull it first:  huggingface-cli download abhishek085/spark-s1-4b-v6-nvfp4 --local-dir $DEC_DIR"; exit 1; }
[ -x .venv/bin/jevcontrol ] && . .venv/bin/activate

IMG="${JEVCONTROL_VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
COMMON=(--gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 -e HF_HUB_OFFLINE=1 --label jevcontrol=1)
CACHES=(-v "$HOME/.cache/vllm:/root/.cache/vllm" -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer")
running() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
wait_ready() { until curl -s -m 2 "http://localhost:$1/v1/models" >/dev/null 2>&1; do sleep 4; done; echo "  ready: :$1"; }

if ! running "${PREFIX}gemma-4-e4b"; then
  docker rm -f "${PREFIX}gemma-4-e4b" >/dev/null 2>&1 || true
  echo "starting main LLM ($LLM_REPO) on :8101 ..."
  docker run -d --name "${PREFIX}gemma-4-e4b" "${COMMON[@]}" -p 8101:8000 "${CACHES[@]}" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMG" --model "$LLM_REPO" --served-model-name gemma-4-e4b --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.22 --max-model-len 8192 --max-num-seqs 16 --trust-remote-code >/dev/null
fi
wait_ready 8101
if ! running "${PREFIX}spark-s1"; then
  docker rm -f "${PREFIX}spark-s1" >/dev/null 2>&1 || true
  echo "starting decision model (spark-s1) on :8102 ..."
  docker run -d --name "${PREFIX}spark-s1" "${COMMON[@]}" -p 8102:8000 "${CACHES[@]}" -v "$PWD/$DEC_DIR:/models/spark-s1:ro" \
    "$IMG" --model /models/spark-s1 --served-model-name spark-s1 --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.12 --max-model-len 8192 --max-num-seqs 16 --trust-remote-code >/dev/null
fi
wait_ready 8102
echo "starting JevControl on http://localhost:8600"
exec jevcontrol serve --port 8600
