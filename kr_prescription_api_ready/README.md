# KR Prescription Extract API (Ready-to-run)

## Run locally (Docker recommended)
```bash
docker build -t kr-prescription-api .
docker run --rm -p 8000:8000 kr-prescription-api
```

Open docs:
- http://localhost:8000/docs

## Run locally (without Docker)
1) Install Tesseract (with Korean language pack)
- Ubuntu/Debian:
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
- macOS:
  brew install tesseract tesseract-lang
- Windows:
  Install Tesseract, then set env var:
  setx TESSERACT_CMD "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

2) Install Python deps:
```bash
pip install -r requirements.txt
```

3) Start:
```bash
uvicorn prescription_api:app --host 0.0.0.0 --port 8000
```

## API
- GET  /healthz
- POST /v1/prescriptions/extract   (multipart/form-data, key: file)

## Production base URL
https://myproject-production-22db.up.railway.app
