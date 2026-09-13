FROM python:3.10-slim

WORKDIR /app

# System deps for torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

RUN useradd -m -u 1000 appuser
RUN chown -R appuser /app
USER appuser

# Render sets the PORT env var at runtime; default to 7860 if not set
# (shell form so ${PORT} actually expands)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
