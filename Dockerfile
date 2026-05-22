FROM python:3.12.7-slim

LABEL maintainer="mlops-submission"
LABEL description="Prediksi Kelulusan Mahasiswa - Model Serving"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY serving/ ./serving/
COPY src/ ./src/
COPY mlruns/ ./mlruns/
COPY MLProject .
COPY conda.yaml .

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run model serving
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
