# Reproducible image for the SSM-causal falsification benchmark.
# Provides a pinned Python + uv environment so `make all` runs end-to-end.
#
#   docker build -t ssm-causal .
#   docker run --rm -v "$PWD/results:/workspace/results" ssm-causal make all
#
# CPU-only by default; for GPU runs of exp24/exp26 mount a CUDA-enabled
# image (e.g. nvidia/cuda:12.1-base) and pip-install torch separately.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    MPLCONFIGDIR=/tmp/matplotlib-cache \
    MPLBACKEND=Agg \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-bibtex-extra \
        biber \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen && uv sync --frozen --extra baselines --extra dev

COPY . .

CMD ["make", "all"]
