# Optional CPU environment for code and evidence checks.
# docker build -t predictive-weights-audit .
# docker run --rm predictive-weights-audit

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
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY ssm_causal ./ssm_causal
RUN uv sync --frozen --extra baselines --extra dev

COPY . .

CMD ["make", "all"]
