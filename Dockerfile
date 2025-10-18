FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ /app/app/
COPY core/ /app/core/
COPY scripts/ /app/scripts/
COPY data/raw/ /app/data/raw/

RUN mkdir -p /app/data/processed


ENV HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/hf-cache/sentence_transformers \
    HF_DATASETS_CACHE=/app/hf-cache/datasets \
    TORCH_HOME=/app/hf-cache/torch

EXPOSE 8000

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_RELOAD=false

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
