# syntax=docker/dockerfile:1.7

# --- build the UI -----------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# vite.config.ts writes the build straight into ../jevcontrol/webui
RUN npm run build

# --- runtime -----------------------------------------------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app

# Only what setuptools needs to install the package, so this layer only rebuilds
# when dependencies or metadata change - not on every source edit.
COPY pyproject.toml README.md ./
COPY jevcontrol/ ./jevcontrol/
COPY --from=frontend /app/jevcontrol/webui ./jevcontrol/webui

RUN pip install --no-cache-dir .

# JEVCONTROL_HOME holds experiments/results/state; JEVCONTROL_MODELS is where
# "Pull" downloads weights to. Mount both as volumes to persist them, or point
# JEVCONTROL_MODELS at an existing models/ directory on the host.
ENV JEVCONTROL_HOME=/data \
    JEVCONTROL_MODELS=/models
RUN useradd -m -u 1000 jevcontrol \
    && mkdir -p /data /models \
    && chown -R jevcontrol:jevcontrol /data /models /app
VOLUME ["/data", "/models"]
USER jevcontrol

# Serving a decision/main-LLM model *from* the Models page starts a Docker
# container on the host and needs Linux + an NVIDIA GPU + the host's Docker
# socket - none of which this image has. Inside a container, point JevControl
# at an OpenAI-compatible endpoint you already run (vLLM, Ollama, a hosted
# API); everything else - Import, review, run, results - works as-is.
EXPOSE 8600
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8600/', timeout=3)" || exit 1

ENTRYPOINT ["jevcontrol", "serve", "--host", "0.0.0.0"]
