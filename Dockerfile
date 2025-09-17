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
RUN apt-get -y install git
WORKDIR /app
COPY cp-hgmm/pyproject.toml cp-hgmm/poetry.loc[k] ./
# RUN poetry config virtualenvs.in-project true
RUN poetry install --no-interaction --no-ansi --no-cache --no-root
ENTRYPOINT []
CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]

FROM base AS prod
RUN apt-get update
RUN apt-get -y install git
RUN apt-get clean
WORKDIR /app/wastebin-mass-estimator
COPY cp-hgmm/pyproject.toml cp-hgmm/poetry.loc[k] ./
RUN poetry install --without dev --no-interaction --no-ansi --no-cache
COPY . .

# Run cron on container startup
CMD ["./start.sh"]