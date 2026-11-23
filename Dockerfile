FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1 \
    PORT=8000

WORKDIR /app

# Install minimal system libs needed by TensorFlow/NumPy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deployment dependencies
COPY deploy/requirements-deploy.txt /app/deploy/requirements-deploy.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/deploy/requirements-deploy.txt

# Copy trained models (untouched from main) and deploy application
COPY model_baseline_LSTM.keras /app/model_baseline_LSTM.keras
COPY model_seq2seq_LSTM.keras /app/model_seq2seq_LSTM.keras
COPY deploy /app/deploy

EXPOSE 8000

CMD ["sh", "-c", "uvicorn deploy.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
