FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    EVALUATOR_WORK_DIR=/data/evaluator_jobs \
    MPLCONFIGDIR=/tmp/matplotlib \
    ULTRALYTICS_CONFIG_DIR=/tmp/ultralytics \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser evaluator_service ./evaluator_service
COPY --chown=appuser:appuser scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY --chown=appuser:appuser q1 ./q1
COPY --chown=appuser:appuser q2 ./q2

RUN mkdir -p /data/evaluator_jobs /tmp/matplotlib /tmp/ultralytics \
    && chown -R appuser:appuser /data /tmp/matplotlib /tmp/ultralytics

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn evaluator_service.main:app --host 0.0.0.0 --port ${PORT}"]
