# prescription_api.py
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_LANG = os.getenv("OCR_LANG", "kor+eng")

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="KR Prescription Extract API",
    version="4.0.0",
    servers=([{"url": PUBLIC_BASE_URL.rstrip("/")}] if PUBLIC_BASE_URL else []),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class Medicine(BaseModel):
    name: str
    dose: Optional[str] = None
    times_per_day: Optional[int] = None
    days: Optional[int] = None
    raw: Optional[str] = None


class PrescriptionExtractResponse(BaseModel):
    success: bool
    text: str
    patient_name: Optional[str] = None
    hospital: Optional[str] = None
    doctor: Optional[str] = None
    issued_date: Optional[str] = None
    medicines: List[Medicine]
    warnings: List[str] = []


# -----------------------------------------------------------------------------
# Image Preprocessing (OCR 극대화)
# -----------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image):
    import numpy as np
    import cv2
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resize_if_small(bgr):
    import cv2
    h, w = bgr.shape[:2]
    if max(h, w) < 1400:
        scale = 1400 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return bgr


def _deskew(bgr):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = 255 - thresh

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return bgr

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 15:
        return bgr

    (h, w) = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _to_pil_gray(np_img):
    return Image.fromarray(np_img).convert("L")


def preprocess_variants(bgr) -> List[Image.Image]:
    import cv2
    import numpy as np

    bgr = _resize_if_small(bgr)
    bgr = _deskew(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.2, (8, 8))

    # Variant 1: Adaptive threshold
    v1 = clahe.apply(gray)
    v1 = cv2.fastNlMeansDenoising(v1, h=12)
    v1 = cv2.adaptiveThreshold(v1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)

    # Variant 2: OTSU
    v2 = clahe.apply(gray)
    v2 = cv2.fastNlMeansDenoising(v2, h=10)
    v2 = cv2.threshold(v2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Variant 3: CLAHE only
    v3 = clahe.apply(gray)

    return [_to_pil_gray(v1), _to_pil_gray(v2), _to_pil_gray(v3)]


# -----------------------------------------------------------------------------
# OCR (멀티 시도 → 최적 결과 선택)
# -----------------------------------------------------------------------------

def _score_text(text: str) -> int:
    if not text:
        return 0
    kor = len(re.findall(r"[가-힣]", text))
    num = len(re.findall(r"\d", text))
    length = len(text)
    noise = len(re.findall(r"[^가-힣A-Za-z0-9\s]", text))
    return length + kor * 3 + num - noise


def _ocr_blocking(pil_img: Image.Image) -> str:
    import pytesseract

    bgr = _pil_to_bgr(pil_img)
    variants = preprocess_variants(bgr)

    best_text = ""
    best_score = 0
    configs = ["--oem 3 --psm 6", "--oem 3 --psm 11"]

    for v in variants:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(v, lang=OCR_LANG, config=cfg).strip()
            except Exception:
                continue
            score = _score_text(txt)
            if score > best_score:
                best_score = score
                best_text = txt

    return best_text.strip()


# -----------------------------------------------------------------------------
# Parsing (후보라도 무조건 생성)
# -----------------------------------------------------------------------------

def _clean_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _parse(text: str) -> Tuple[Dict[str, Optional[str]], List[Medicine]]:
    t = _clean_text(text)
    lines = [l.strip() for l in t.split("\n") if l.strip()]

    meds: List[Medicine] = []
    dose_pat = re.compile(r"(\d+)\s*(정|캡슐|포|mg|㎎|ml|mL)", re.I)

    for ln in lines:
        if any(k in ln for k in ["복용", "식후", "아침", "저녁", "1일", "회"]):
            dm = dose_pat.search(ln)
            meds.append(Medicine(
                name=ln[:60],
                dose=dm.group(0) if dm else None,
                raw=ln
            ))

    if not meds and lines:
        meds.append(Medicine(
            name=lines[0][:60],
            raw=lines[0]
        ))

    meta = {
        "patient_name": None,
        "hospital": None,
        "doctor": None,
        "issued_date": None,
    }

    return meta, meds


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/prescriptions/extract", response_model=PrescriptionExtractResponse)
async def extract_prescription(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    text = await run_in_threadpool(_ocr_blocking, pil_img)
    if not text:
        raise HTTPException(status_code=422, detail="OCR failed to extract text")

    meta, meds = await run_in_threadpool(_parse, text)

    return PrescriptionExtractResponse(
        success=True,
        text=text,
        patient_name=meta["patient_name"],
        hospital=meta["hospital"],
        doctor=meta["doctor"],
        issued_date=meta["issued_date"],
        medicines=meds,
        warnings=[],
    )
