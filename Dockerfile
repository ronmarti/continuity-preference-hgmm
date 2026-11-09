FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS base

LABEL version="2025.8" maintainer="https://github.com/ronmarti"

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=2.1

RUN pip install pipx && \
    pipx ensurepath && \
    pipx install "poetry==$POETRY_VERSION"
# Ensure poetry is in PATH
ENV PATH="/root/.local/bin:$PATH"
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc

FROM base AS dev
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /cp-gmm-app
COPY cp-hgmm/pyproject.toml cp-hgmm/poetry.loc[k] ./
# RUN poetry config virtualenvs.in-project true
RUN poetry install --no-interaction --no-ansi --no-cache --no-root && \
    ln -s "$(poetry env info --path)" /opt/cp-hgmm-venv
ENTRYPOINT []
CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]

FROM base AS prod
WORKDIR /cp-gmm-app/cp-hgmm
COPY cp-hgmm/pyproject.toml cp-hgmm/poetry.loc[k] ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root && \
    ln -s "$(poetry env info --path)" /opt/cp-hgmm-venv
COPY cp-hgmm/ ./
ENV PATH="/opt/cp-hgmm-venv/bin:$PATH" \
    PYTHONPATH=/cp-gmm-app/cp-hgmm \
    MPLBACKEND=Agg \
    CPHGMM_RUNS_DIR=/data/runs
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/')" || exit 1
# A single worker keeps one shared training job visible to every browser session.
CMD ["solara", "run", "gui/app.py", "--host", "0.0.0.0", "--port", "8765", "--production", "--workers", "1", "--no-open"]