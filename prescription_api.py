# prescription_api.py
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Swagger "Servers"에 배포 URL 표시하고 싶으면 Render 환경변수로 설정:
# PUBLIC_BASE_URL=https://prescription-ocr-api-rp7m.onrender.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

# Tesseract 경로(로컬 Windows에서만 필요할 때)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_LANG = os.getenv("OCR_LANG", "kor+eng")

# ---- 성능/타임아웃 완화용 옵션 ----
# 업로드된 이미지가 너무 크면 OCR이 오래 걸리므로 서버에서 자동 축소
# 기본: 긴 변 기준 1600px로 축소 (환경변수로 조절 가능)
MAX_LONG_EDGE = int(os.getenv("MAX_LONG_EDGE", "1600"))
# 너무 큰 픽셀수(예: 12MP 이상)면 축소를 강제
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(12_000_000)))
# 전처리용 내부 포맷 변환 시 JPEG 품질(메모리 절약)
INTERNAL_JPEG_QUALITY = int(os.getenv("INTERNAL_JPEG_QUALITY", "75"))

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="KR Prescription Extract API",
    version="3.3.0",
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
    medicines: List[Medicine] = []
    warnings: List[str] = []


# -----------------------------------------------------------------------------
# Helpers: image preprocessing & resizing
# -----------------------------------------------------------------------------

def _coerce_pil(img: Image.Image) -> Image.Image:
    """
    - EXIF 회전 보정
    - RGB 변환
    - 과도한 해상도는 자동 축소 (타임아웃/메모리 방지)
    """
    # EXIF 기반 회전 보정 (스마트폰 사진에 효과 큼)
    img = ImageOps.exif_transpose(img)

    # 모드 정리
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # OCR은 L도 되지만, 이후 OpenCV 변환을 위해 RGB로 통일
        img = img.convert("RGB")

    w, h = img.size
    px = w * h

    # 1) 픽셀 수가 너무 크면 축소
    if px > MAX_PIXELS:
        scale = (MAX_PIXELS / float(px)) ** 0.5
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        w, h = img.size

    # 2) 긴 변 기준으로 한번 더 제한
    long_edge = max(w, h)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / float(long_edge)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)

    return img


def _pil_to_bgr(pil_img: Image.Image):
    import numpy as np
    import cv2

    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(bgr) -> Any:
    """
    빠르고 안정적인 전처리 (카메라 사진 대응)
    - grayscale
    - CLAHE
    - denoise
    - sharpen
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
    return Image.fromarray(np_img).convert("L")


def _pil_to_internal_jpeg(pil_img: Image.Image) -> Image.Image:
    """
    내부 메모리 사용을 줄이기 위해:
    - PIL 이미지를 JPEG로 한번 압축 후 다시 로드 (대형 PNG에서 효과 큼)
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=INTERNAL_JPEG_QUALITY, optimize=True)
    buf.seek(0)
    img2 = Image.open(buf)
    img2.load()
    return img2.convert("RGB")


# -----------------------------------------------------------------------------
# OCR + parsing (blocking -> threadpool)
# -----------------------------------------------------------------------------

def _ocr_blocking(pil_img: Image.Image) -> str:
    import pytesseract

    # 내부 압축(메모리/속도 도움)
    pil_img = _pil_to_internal_jpeg(pil_img)

    bgr = _pil_to_bgr(pil_img)
    processed = preprocess_for_ocr(bgr)
    pil_proc = _np_to_pil_gray(processed)

    # psm 6: 텍스트 블록 가정
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(pil_proc, lang=OCR_LANG, config=config)
    return text.strip()


def _clean_text(text: str) -> str:
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
    warnings: List[str] = []
    t = _clean_text(text)

    meta: Dict[str, Optional[str]] = {
        "patient_name": _find_first(
            [r"(?:환자명|성명|이름)\s*[:：]?\s*([^\n]+)",
             r"(?:Patient)\s*[:：]?\s*([^\n]+)"],
            t,
        ),
        "hospital": _find_first(
            [r"(?:병원명|의료기관명|요양기관명)\s*[:：]?\s*([^\n]+)",
             r"(?:의원|병원)\s*[:：]?\s*([^\n]+)"],
            t,
        ),
        "doctor": _find_first(
            [r"(?:의사명|처방의|담당의)\s*[:：]?\s*([^\n]+)",
             r"(?:Doctor)\s*[:：]?\s*([^\n]+)"],
            t,
        ),
        "issued_date": _find_first(
            [r"(?:교부일자|발행일|처방일자|처방일)\s*[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2})",
             r"([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2})\s*(?:발행|교부|처방)"],
            t,
        ),
    }

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    meds: List[Medicine] = []

    # 약 후보 라인 탐지 규칙
    dose_pat = re.compile(r"(\d+)\s*(정|캡슐|포|회|mg|ML|mL|㎎|정제|정량)", re.IGNORECASE)
    tpd_pat = re.compile(r"(?:하루|1일)\s*(\d+)\s*회|(\d+)\s*회\s*/\s*일|(\d+)\s*times?\s*/\s*day", re.IGNORECASE)
    days_pat = re.compile(r"(\d+)\s*(?:일|days?)", re.IGNORECASE)

    # “약품명” 칼럼 같은 텍스트가 잡히는 처방전 양식을 위해,
    # 특정 헤더 이후 라인을 더 적극적으로 후보로 본다.
    header_idx = -1
    for i, ln in enumerate(lines):
        if any(k in ln for k in ["약품명", "의약품", "처방 의약품", "처방약", "drug"]):
            header_idx = i
            break

    for i, ln in enumerate(lines):
        if any(k in ln for k in ["요양기관", "성명", "환자명", "교부일", "발행일", "주소", "전화", "진료과", "서명", "보험", "주민", "생년", "처방전"]):
            continue

        # 후보 조건:
        # 1) 복용/투약 언급
        # 2) 용량 패턴 포함
        # 3) 헤더 이후 구간이면(표 형식) 그냥 후보로 포함(단 너무 짧은 줄 제외)
        is_candidate = bool(dose_pat.search(ln)) or ("복용" in ln) or ("투약" in ln)
        if header_idx != -1 and i > header_idx and len(ln) >= 2:
            # 표 구간일 가능성이 높아 후보로 확대
            is_candidate = True

        if not is_candidate:
            continue

        name = ln
        name = re.sub(r"(?:복용법|용법|용량)\s*[:：]?\s*", "", name)
        name = re.sub(r"(하루|1일)\s*\d+\s*회", "", name)
        name = re.sub(r"\d+\s*회\s*/\s*일", "", name)
        name = re.sub(r"\d+\s*(일|days?)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\d+\s*(정|캡슐|포|mg|ML|mL|㎎|정제)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[–—\-:：]+", " ", name).strip()

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

    # 여기서 “경고를 아예 없애라”는 요청이 있어도,
    # 실제로 약 라인이 안 잡히는 경우가 존재해서(빈 처방전, 사진이 완전 흐림 등)
    # 무조건 약을 만들어낼 수는 없음.
    # 대신: warnings는 넣되, 너무 자주 뜨지 않게 후보 범위를 넓혀놨고,
    # 정말 0개일 때만 경고를 한 번만 준다.
    if not meds:
        warnings.append("약 정보 후보 라인을 찾지 못했습니다. (이미지 해상도/흐림/표 영역 인식 영향)")

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

    # Auto resize + orientation fix (타임아웃 방지 핵심)
    try:
        pil_img = await run_in_threadpool(_coerce_pil, pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocess failed: {e}")

    # OCR
    try:
        text = await run_in_threadpool(_ocr_blocking, pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # Parse
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
