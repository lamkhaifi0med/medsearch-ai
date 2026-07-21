# Hugging Face Spaces single-container image: React build + FastAPI + case data.
# The API serves the built frontend itself (see app/main.py static mount).

# ---- stage 1: frontend build
FROM node:22-alpine AS webbuild
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ .
RUN npm run build

# ---- stage 2: runtime
FROM python:3.12-slim

WORKDIR /srv

# CPU-only torch first (avoids multi-GB CUDA wheels).
RUN pip install --no-cache-dir "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=webbuild /src/dist ./static
COPY data/processed/cases_clean.jsonl ./data/processed/cases_clean.jsonl

# HF Spaces runs as UID 1000 with an ephemeral filesystem: caches go to /tmp.
ENV HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    CASES_FILE=/srv/data/processed/cases_clean.jsonl

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
