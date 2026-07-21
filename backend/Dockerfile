# CPU-only backend image. BGE-M3 weights are NOT baked in — they live in a
# named volume mounted at /root/.cache/huggingface (downloaded on first boot).
FROM python:3.12-slim

WORKDIR /srv

# CPU-only torch first (avoids multi-GB CUDA wheels).
# >=2.6 required: transformers v5 refuses torch.load on older versions (CVE-2025-32434).
RUN pip install --no-cache-dir "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
