FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for opencv + tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY prescription_api.py /app/prescription_api.py

# Railway provides $PORT. Locally defaults to 8000.
CMD ["sh", "-c", "uvicorn prescription_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
