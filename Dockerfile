FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PORT=8000

WORKDIR /app

# Install lightweight deployment dependencies (no TensorFlow runtime needed)
COPY deploy/requirements-deploy.txt /app/deploy/requirements-deploy.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/deploy/requirements-deploy.txt

# Copy trained .keras weight archives (untouched from main) and deploy app
COPY model_baseline_LSTM.keras /app/model_baseline_LSTM.keras
COPY model_seq2seq_LSTM.keras /app/model_seq2seq_LSTM.keras
COPY deploy /app/deploy

EXPOSE 8000

CMD ["sh", "-c", "uvicorn deploy.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
