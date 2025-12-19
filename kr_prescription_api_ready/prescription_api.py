# -*- coding: utf-8 -*-
"""
Korean Standard Prescription OCR API (Railway-ready)

- Endpoint:
    POST /v1/prescriptions/extract   (multipart form, field name: file)
- Health:
    GET  /healthz
- Docs:
    /docs

Deployment notes (Railway):
- Start command:
    uvicorn prescription_api:app --host 0.0.0.0 --port $PORT
- Make sure Tesseract + Korean language pack are installed in the container:
    apt-get install -y tesseract-ocr tesseract-ocr-kor
"""

from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ============================================================
# Config
# ============================================================
# Your deployed host (used for OpenAPI "servers" and debug metadata)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://myproject-production-22db.up.railway.app/").rstrip("/") + "/"

# Windows support (optional). Leave empty in Linux containers.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="KR Prescription Extract API",
    version="3.1.0",
    servers=[{"url": PUBLIC_BASE_URL.rstrip("/")}],  # shows your Railway host in Swagger
)

# CORS (so mobile/web apps can call the API directly)
# In production, you can restrict origins instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Models
# ============================================================
class IntakeInstruction(BaseModel):
    dose_per_time: Optional[str] = None      # "1정", "5mL", "500mg", or "1" (if unit missing)
    times_per_day: Optional[int] = None      # 3
    days: Optional[int] = None               # 5
    timing: List[str] = []                   # ["식후","취침전","PRN"...]
    free_text: Optional[str] = None          # raw row/line


class MedicineItem(BaseModel):
    name: str
    instruction: IntakeInstruction
    raw_row_text: Optional[str] = None


class PrescriptionResponse(BaseModel):
    raw_text: str
    medicines: List[MedicineItem]
    global_hint: Optional[IntakeInstruction] = None
    debug: Optional[Dict[str, Any]] = None


# ============================================================
# Regex / constants
# ============================================================
FORMS = ["정", "캡슐", "시럽", "산", "액", "연고", "크림", "패치", "드롭", "현탁액", "주", "주사"]
NOISE = ["처방전", "처방", "투약", "복용", "용법", "용량", "환자", "병원", "약국", "진단", "성명", "생년", "보험", "의사", "서명"]

STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|ug|mL|ml)", re.IGNORECASE)
DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(정|캡슐|포|병|mL|ml|mg|g)", re.IGNORECASE)
TIMES1_RE = re.compile(r"(1일|하루)\s*(\d+)\s*회")
TIMES2_RE = re.compile(r"(\d+)\s*회\s*/\s*(일|day)", re.IGNORECASE)
DAYS_RE = re.compile(r"(\d+)\s*(일|일분|일간|days?)", re.IGNORECASE)

TIMING_RE = re.compile(
    r"(아침|점심|저녁|취침전|자기전|취침|식전|식후|공복|필요시|PRN|통증시|열나면|증상시|BID|TID|QID)",
    re.IGNORECASE
)

STOP_TOKENS = ["합계", "총", "의사", "서명", "병원", "약국", "환자", "발행", "보험", "진단"]


# ============================================================
# Utility
# ============================================================
def normalize_timing(token: str) -> str:
    t = token.lower()
    if "취침" in t or "자기전" in t:
        return "취침전"
    if "식전" in t:
        return "식전"
    if "식후" in t:
        return "식후"
    if "prn" in t or "필요시" in t or "통증시" in t or "열나면" in t or "증상시" in t:
        return "PRN"
    if t == "bid":
        return "BID(하루2회)"
    if t == "tid":
        return "TID(하루3회)"
    if t == "qid":
        return "QID(하루4회)"
    return token


def key(s: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", re.sub(r"\s+", "", s.lower()))


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Camera photo-friendly preprocessing:
    - CLAHE contrast
    - denoise
    - mild sharpen
    - adaptive threshold
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    den = cv2.fastNlMeansDenoising(gray, h=12)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(den, -1, kernel)

    thr = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )
    return thr


def run_ocr_text(bin_img: np.ndarray) -> str:
    config = "--oem 3 --psm 6 -l kor+eng"
    text = pytesseract.image_to_string(bin_img, config=config)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ============================================================
# OCR table helpers (word-box based)
# ============================================================
def ocr_words(bin_img: np.ndarray) -> pd.DataFrame:
    config = "--oem 3 --psm 6 -l kor+eng"
    df = pytesseract.image_to_data(bin_img, config=config, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["conf"].astype(float) > 0)]
    return df[["text", "left", "top", "width", "height", "conf"]].copy()


def cluster_rows(words: pd.DataFrame) -> List[pd.DataFrame]:
    if words.empty:
        return []
    h_med = float(words["height"].median())
    tol = max(8.0, h_med * 0.65)

    words = words.sort_values(["top", "left"]).reset_index(drop=True)

    rows: List[pd.DataFrame] = []
    current: List[pd.Series] = []
    current_y: Optional[float] = None

    for _, r in words.iterrows():
        y = float(r["top"])
        if current_y is None:
            current_y = y
            current.append(r)
        elif abs(y - current_y) <= tol:
            current.append(r)
            current_y = current_y * 0.8 + y * 0.2
        else:
            if len(current) >= 2:
                rows.append(pd.DataFrame(current))
            current = [r]
            current_y = y

    if current and len(current) >= 2:
        rows.append(pd.DataFrame(current))

    return rows


def detect_header_and_boundaries(rows: List[pd.DataFrame]) -> Tuple[Optional[int], List[int]]:
    """
    Find header row and infer column boundaries for typical KR standard prescription table.
    """
    header_keywords = ["의약", "약품", "품명", "1회", "투약", "투여", "1일", "횟수", "일수"]
    best_idx, best_score = None, -1

    for i, df in enumerate(rows[:12]):
        line = " ".join(df["text"].tolist())
        score = sum(1 for k in header_keywords if k in line)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None or best_score < 2:
        return None, []

    header = rows[best_idx].sort_values("left")
    xs = sorted(header["left"].astype(int).tolist())
    if len(xs) < 4:
        return best_idx, []

    gaps = [(b - a, a, b) for a, b in zip(xs, xs[1:])]
    gaps.sort(reverse=True, key=lambda x: x[0])

    boundaries: List[int] = []
    for g, a, b in gaps:
        if g > 40:
            boundaries.append((a + b) // 2)
        if len(boundaries) >= 3:
            break

    return best_idx, sorted(set(boundaries))


def split_row(row_df: pd.DataFrame, boundaries: List[int]) -> List[str]:
    row_df = row_df.sort_values("left")
    cols: List[List[str]] = [[] for _ in range(len(boundaries) + 1)]

    for _, r in row_df.iterrows():
        x = int(r["left"])
        idx = 0
        while idx < len(boundaries) and x > boundaries[idx]:
            idx += 1
        cols[idx].append(str(r["text"]))

    return [re.sub(r"\s+", " ", " ".join(c)).strip() for c in cols]


# ============================================================
# Extraction logic
# ============================================================
def extract_global_hint(text: str) -> IntakeInstruction:
    dose = None
    m = DOSE_RE.search(text)
    if m:
        dose = f"{m.group(1)}{m.group(2)}"

    times = None
    m1 = TIMES1_RE.search(text)
    if m1:
        times = int(m1.group(2))
    else:
        m2 = TIMES2_RE.search(text)
        if m2:
            times = int(m2.group(1))

    days = None
    ds = [int(m.group(1)) for m in DAYS_RE.finditer(text)]
    if ds:
        days = max(ds)

    timing = list(dict.fromkeys(normalize_timing(m.group(1)) for m in TIMING_RE.finditer(text)))

    return IntakeInstruction(dose_per_time=dose, times_per_day=times, days=days, timing=timing)


def parse_int(cell: str) -> Optional[int]:
    m = re.search(r"\d+", cell or "")
    return int(m.group(0)) if m else None


def parse_table(bin_img: np.ndarray, raw_text: str) -> Tuple[List[MedicineItem], IntakeInstruction, Dict[str, Any]]:
    words = ocr_words(bin_img)
    rows = cluster_rows(words)
    header_idx, boundaries = detect_header_and_boundaries(rows)

    global_hint = extract_global_hint(raw_text)

    debug: Dict[str, Any] = {
        "mode": "table",
        "rows_detected": len(rows),
        "header_idx": header_idx,
        "boundaries": boundaries,
    }

    if header_idx is None or not boundaries:
        debug["mode"] = "fallback_needed"
        return [], global_hint, debug

    medicines: List[MedicineItem] = []
    for r in rows[header_idx + 1 :]:
        cols = split_row(r, boundaries)
        row_text = " | ".join(cols).strip()
        if not row_text:
            continue
        if any(tok in row_text for tok in STOP_TOKENS):
            continue

        name = cols[0] if len(cols) > 0 else ""
        if len(name) < 2 or any(n in name for n in NOISE):
            continue

        dose_cell = cols[1] if len(cols) > 1 else ""
        times_cell = cols[2] if len(cols) > 2 else ""
        days_cell = cols[3] if len(cols) > 3 else ""

        # dose
        dose = None
        m = DOSE_RE.search(dose_cell)
        if m:
            dose = f"{m.group(1)}{m.group(2)}"
        else:
            dose = dose_cell.strip() or None

        times = parse_int(times_cell)
        days = parse_int(days_cell)

        inst = IntakeInstruction(
            dose_per_time=dose or global_hint.dose_per_time,
            times_per_day=times or global_hint.times_per_day,
            days=days or global_hint.days,
            timing=global_hint.timing,
            free_text=row_text
        )

        medicines.append(MedicineItem(name=name, instruction=inst, raw_row_text=row_text))

    # dedupe by medicine name
    dedup: List[MedicineItem] = []
    seen = set()
    for m in medicines:
        k = key(m.name)
        if k and k not in seen:
            seen.add(k)
            dedup.append(m)

    return dedup, global_hint, debug


def fallback_line_parse(raw_text: str, global_hint: IntakeInstruction) -> List[MedicineItem]:
    """
    Minimal fallback when table/header detection fails.
    """
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    meds: List[MedicineItem] = []
    seen = set()

    def score(line: str) -> int:
        if any(n in line for n in NOISE):
            return 0
        s = 0
        if any(f in line for f in FORMS):
            s += 4
        if STRENGTH_RE.search(line):
            s += 2
        if re.search(r"[A-Za-z가-힣]", line):
            s += 1
        if "식후" in line or "식전" in line or "하루" in line or "1일" in line:
            s -= 1
        if sum(ch.isdigit() for ch in line) >= 10:
            s -= 2
        return s

    scored = sorted([(ln, score(ln)) for ln in lines], key=lambda x: x[1], reverse=True)
    candidates = [ln for ln, sc in scored if sc > 0][:30]

    for ln in candidates:
        # remove dosage-like tokens to keep the name
        name = ln
        name = STRENGTH_RE.sub(" ", name)
        name = DOSE_RE.sub(" ", name)
        name = TIMES1_RE.sub(" ", name)
        name = TIMES2_RE.sub(" ", name)
        name = DAYS_RE.sub(" ", name)
        name = TIMING_RE.sub(" ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if len(name) < 3:
            continue

        k = key(name)
        if not k or k in seen:
            continue
        seen.add(k)

        # local instruction
        dose = None
        md = DOSE_RE.search(ln)
        if md:
            dose = f"{md.group(1)}{md.group(2)}"

        times = None
        mt = TIMES1_RE.search(ln)
        if mt:
            times = int(mt.group(2))
        else:
            mt2 = TIMES2_RE.search(ln)
            if mt2:
                times = int(mt2.group(1))

        ds = [int(m.group(1)) for m in DAYS_RE.finditer(ln)]
        days = max(ds) if ds else None

        timing = list(dict.fromkeys(normalize_timing(m.group(1)) for m in TIMING_RE.finditer(raw_text)))

        inst = IntakeInstruction(
            dose_per_time=dose or global_hint.dose_per_time,
            times_per_day=times or global_hint.times_per_day,
            days=days or global_hint.days,
            timing=timing or global_hint.timing,
            free_text=ln
        )

        meds.append(MedicineItem(name=name, instruction=inst, raw_row_text=ln))

    return meds


# ============================================================
# Routes
# ============================================================
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "service": "KR Prescription Extract API", "base_url": PUBLIC_BASE_URL}


@app.post("/v1/prescriptions/extract", response_model=PrescriptionResponse)
async def extract_prescription(file: UploadFile = File(...)) -> PrescriptionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")

    image_bytes = await file.read()
    bgr = bytes_to_bgr(image_bytes)
    bin_img = preprocess_for_ocr(bgr)

    raw_text = run_ocr_text(bin_img)

    medicines, global_hint, debug = parse_table(bin_img, raw_text)

    if not medicines:
        medicines = fallback_line_parse(raw_text, global_hint)
        debug["mode"] = "fallback_line"

    return PrescriptionResponse(
        raw_text=raw_text,
        medicines=medicines,
        global_hint=global_hint,
        debug={
            "public_base_url": PUBLIC_BASE_URL,
            "filename": file.filename,
            "content_type": file.content_type,
            **debug
        }
    )
