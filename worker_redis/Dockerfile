FROM python:3.10-slim

LABEL maintainer="ProctorSystem"
LABEL description="ML Worker for real-time anti-cheating (Redis-based)"

WORKDIR /app

# System dependencies for OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (ML + Redis only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code
COPY worker.py .
COPY model_engine.py .

# Models are mounted via Docker volume at /app/
# best.pt, yolov8n.pt, face_landmarker.task must exist at /app/

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MODELS_DIR=/app

# One worker process per container — scale via docker compose --scale worker=N
CMD ["python", "worker.py"]
