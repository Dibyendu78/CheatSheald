FROM python:3.10-slim

LABEL maintainer="ProctorSystem"
LABEL description="FastAPI WebSocket API for real-time anti-cheating system"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY main.py .

# Health check via HTTP
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
