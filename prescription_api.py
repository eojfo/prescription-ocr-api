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

# IMPORTANT:
# - Default must NOT point to an old host (Railway). Empty by default.
# - If you want Swagger "Servers" to show your deployed host, set env var:
#   PUBLIC_BASE_URL=https://prescription-ocr-api-rp7m.onrender.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

# Optional: Windows local override (leave empty in Linux containers like Render)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    import pytesseract  # safe import
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# OCR language
OCR_LANG = os.getenv("OCR_LANG", "kor+eng")

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="KR Prescription Extract API",
    version="3.2.0",
    # Only include servers when PUBLIC_BASE_URL is set (prevents Swagger/OpenAPI issues)
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
    dose: Optional[str] = None           # e.g., "1정", "2캡슐"
    times_per_day: Optional[int] = None  # e.g., 3
    days: Optional[int] = None           # e.g., 5
    raw: Optional[str] = None            # raw line text for debugging


class PrescriptionExtractResponse(BaseModel):
    success: bool
    text: str
    patient_name: Optional[str] = None
    hospital: Optional[str] = None
    doctor: Optional[str] = None
    issued_date: Optional[str] = None
    medicines: List[Medicine] = []
    warnings: List[str] = []


# -----------------------------------------------------------------------------
# Helpers: image preprocessing (kept fast; no blocking network)
# -----------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image):
    # Lazy import to reduce startup overhead and avoid import-time issues
    import numpy as np
    import cv2

    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(bgr) -> Any:
    """
    Camera photo-friendly preprocessing:
    - CLAHE contrast
    - denoise
    - mild sharpen
    - adaptive threshold
    """
    import numpy as np
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    den = cv2.fastNlMeansDenoising(gray, h=12)

    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(den, -1, kernel)

    thr = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 9
    )
    return thr


def _np_to_pil_gray(np_img) -> Image.Image:
    # np_img is a single-channel uint8 image
    return Image.fromarray(np_img).convert("L")


# -----------------------------------------------------------------------------
# Helpers: OCR + parsing (blocking -> must run in threadpool)
# -----------------------------------------------------------------------------

def _ocr_blocking(pil_img: Image.Image) -> str:
    """
    BLOCKING OCR function. Must not run directly in async context.
    """
    import pytesseract

    bgr = _pil_to_bgr(pil_img)
    processed = preprocess_for_ocr(bgr)
    pil_proc = _np_to_pil_gray(processed)

    # config tuned a bit for mixed KR/EN
    # psm 6 = assume a block of text
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(pil_proc, lang=OCR_LANG, config=config)
    return text.strip()


def _clean_text(text: str) -> str:
    # normalize whitespace
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_first(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            return val if val else None
    return None


def _parse_prescription_blocking(text: str) -> Tuple[Dict[str, Optional[str]], List[Medicine], List[str]]:
    """
    Very lightweight heuristic parser for KR prescription-like text.
    This is best-effort because formats vary by hospital/pharmacy.
    """
    warnings: List[str] = []

    t = _clean_text(text)

    meta: Dict[str, Optional[str]] = {
        "patient_name": _find_first(
            [
                r"(?:환자명|성명|이름)\s*[:：]?\s*([^\n]+)",
                r"(?:Patient)\s*[:：]?\s*([^\n]+)",
            ],
            t,
        ),
        "hospital": _find_first(
            [
                r"(?:병원명|의료기관명|요양기관명)\s*[:：]?\s*([^\n]+)",
                r"(?:의원|병원)\s*[:：]?\s*([^\n]+)",
            ],
            t,
        ),
        "doctor": _find_first(
            [
                r"(?:의사명|처방의|담당의)\s*[:：]?\s*([^\n]+)",
                r"(?:Doctor)\s*[:：]?\s*([^\n]+)",
            ],
            t,
        ),
        "issued_date": _find_first(
            [
                r"(?:교부일자|발행일|처방일자|처방일)\s*[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2})",
                r"([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2})\s*(?:발행|교부|처방)",
            ],
            t,
        ),
    }

    # Medicine line heuristics:
    # Look for lines that include dose patterns (정/캡슐/포/밀리/정제 etc) or "복용" patterns
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    meds: List[Medicine] = []

    dose_pat = re.compile(r"(\d+)\s*(정|캡슐|포|회|mg|ML|mL|㎎|정제|정량)", re.IGNORECASE)
    tpd_pat = re.compile(r"(?:하루|1일)\s*(\d+)\s*회|(\d+)\s*회\s*/\s*일|(\d+)\s*times?\s*/\s*day", re.IGNORECASE)
    days_pat = re.compile(r"(\d+)\s*(?:일|days?)", re.IGNORECASE)

    for ln in lines:
        # skip obvious non-med lines
        if any(k in ln for k in ["요양기관", "성명", "환자명", "교부일", "발행일", "주소", "전화", "진료과", "서명", "보험", "주민", "생년", "처방전"]):
            continue

        # candidate if contains dose or mentions 복용/투약
        is_candidate = bool(dose_pat.search(ln)) or ("복용" in ln) or ("투약" in ln)
        if not is_candidate:
            continue

        # Try to extract medicine name: remove common trailing instruction tokens
        name = ln
        # remove dose/times/days fragments to isolate a name-ish part
        name = re.sub(r"(?:복용법|용법|용량)\s*[:：]?\s*", "", name)
        name = re.sub(r"(하루|1일)\s*\d+\s*회", "", name)
        name = re.sub(r"\d+\s*회\s*/\s*일", "", name)
        name = re.sub(r"\d+\s*(일|days?)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\d+\s*(정|캡슐|포|mg|ML|mL|㎎|정제)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[–—\-:：]+", " ", name).strip()

        # If name becomes too short, fallback to original line
        if len(name) < 2:
            name = ln.strip()

        dose = None
        dm = dose_pat.search(ln)
        if dm:
            dose = f"{dm.group(1)}{dm.group(2)}"

        times_per_day = None
        tm = tpd_pat.search(ln)
        if tm:
            for g in tm.groups():
                if g:
                    try:
                        times_per_day = int(g)
                        break
                    except:
                        pass

        days = None
        dsm = days_pat.search(ln)
        if dsm:
            try:
                days = int(dsm.group(1))
            except:
                pass

        meds.append(Medicine(name=name, dose=dose, times_per_day=times_per_day, days=days, raw=ln))

    if not meds:
        warnings.append("약 정보(약품명/복용법) 후보 라인을 찾지 못했습니다. (처방전 양식/사진 품질/기울기 영향)")

    return meta, meds, warnings


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/prescriptions/extract", response_model=PrescriptionExtractResponse)
async def extract_prescription(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    # Basic content-type check (some cameras may send octet-stream; allow it)
    if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported content_type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Decode image
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # OCR (blocking -> threadpool)
    try:
        text = await run_in_threadpool(_ocr_blocking, pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # Parse (blocking -> threadpool)
    meta, meds, warnings = await run_in_threadpool(_parse_prescription_blocking, text)

    return PrescriptionExtractResponse(
        success=True,
        text=text,
        patient_name=meta.get("patient_name"),
        hospital=meta.get("hospital"),
        doctor=meta.get("doctor"),
        issued_date=meta.get("issued_date"),
        medicines=meds,
        warnings=warnings,
    )
