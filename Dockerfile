FROM rocm/pytorch:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY training/ ./training/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.main"]
