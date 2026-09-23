# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Heart Attack Prediction API (FastAPI)
# Target: IBM Cloud Code Engine / IBM Cloud Foundry / any OCI-compatible runtime
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: base image ──────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent .pyc files and enable unbuffered stdout for container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# ── Stage 2: dependency install ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ── Stage 3: copy project files ───────────────────────────────────────────────
COPY data/       ./data/
COPY model/      ./model/
COPY src/        ./src/
COPY app.py      .
COPY train.py    .

# Ensure src is importable as a package
ENV PYTHONPATH=/app

# ── Stage 4: optional model pre-build ────────────────────────────────────────
# If model/pipeline.pkl is not already present (e.g. in a fresh CI build),
# train the model at image-build time.  Comment this out if you mount the
# model artefacts as a volume or object-storage bucket at runtime.
RUN if [ ! -f /app/model/pipeline.pkl ]; then python train.py; fi

# ── Stage 5: runtime ─────────────────────────────────────────────────────────
EXPOSE ${PORT}

# Use Gunicorn + Uvicorn worker for production throughput.
# IBM Code Engine honours the $PORT env var; default is 8000.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 2"]
