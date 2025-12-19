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

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_LANG = os.getenv("OCR_LANG", "kor+eng")

# Performance / timeout reduction
MAX_LONG_EDGE = int(os.getenv("MAX_LONG_EDGE", "1800"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(14_000_000)))
INTERNAL_JPEG_QUALITY = int(os.getenv("INTERNAL_JPEG_QUALITY", "78"))

# Table detection
MIN_TABLE_AREA_RATIO = float(os.getenv("MIN_TABLE_AREA_RATIO", "0.08"))
MAX_TABLE_AREA_RATIO = float(os.getenv("MAX_TABLE_AREA_RATIO", "0.95"))
TABLE_BORDER_PAD = int(os.getenv("TABLE_BORDER_PAD", "12"))

# Deskew tuning
DESKEW_MAX_DEG = float(os.getenv("DESKEW_MAX_DEG", "12.0"))  # ignore crazy angles
DESKEW_MIN_CONF = float(os.getenv("DESKEW_MIN_CONF", "0.15"))  # if too weak, skip

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
    dose: Optional[str] = None           # 1회 투약량 (예: 1정, 0.5정, 10mL)
    times_per_day: Optional[int] = None  # 1일 투여횟수 (예: 3)
    days: Optional[int] = None           # 총 투약일수 (예: 5)
    instructions: Optional[str] = None   # 식전/식후/취침전/PRN 등
    raw: Optional[str] = None            # 디버깅용 원문 라인


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
# Helpers: PIL normalize / resize
# -----------------------------------------------------------------------------

def _coerce_pil(img: Image.Image) -> Image.Image:
    # EXIF orientation fix
    img = ImageOps.exif_transpose(img)

    # Normalize mode
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    w, h = img.size
    px = w * h

    # Too many pixels -> shrink by pixel cap
    if px > MAX_PIXELS:
        scale = (MAX_PIXELS / float(px)) ** 0.5
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        w, h = img.size

    # Long edge cap
    long_edge = max(w, h)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / float(long_edge)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)

    return img


def _pil_to_internal_jpeg(pil_img: Image.Image) -> Image.Image:
    """
    Large PNG -> heavy memory; compress internally for speed.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=INTERNAL_JPEG_QUALITY, optimize=True)
    buf.seek(0)
    img2 = Image.open(buf)
    img2.load()
    return img2.convert("RGB")


def _pil_to_bgr(pil_img: Image.Image):
    import numpy as np
    import cv2
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr) -> Image.Image:
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _np_to_pil_gray(np_img) -> Image.Image:
    return Image.fromarray(np_img).convert("L")


# -----------------------------------------------------------------------------
# Preprocess for OCR
# -----------------------------------------------------------------------------

def preprocess_for_ocr(bgr) -> Any:
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


# -----------------------------------------------------------------------------
# Deskew (auto tilt correction)
# -----------------------------------------------------------------------------

def _estimate_skew_angle_deg(bgr) -> Tuple[float, float]:
    """
    Returns (angle_deg, confidence 0..1).
    Uses Hough line angles from binarized edges.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # edges
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120,
        minLineLength=max(80, bgr.shape[1] // 10),
        maxLineGap=15,
    )
    if lines is None or len(lines) < 5:
        return 0.0, 0.0

    angles = []
    weights = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        length = (dx * dx + dy * dy) ** 0.5
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # keep near-horizontal lines (most text lines / table lines)
        # normalize to [-90, 90]
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        # prefer angles within +-45 deg
        if abs(angle) <= 45:
            angles.append(angle)
            weights.append(length)

    if len(angles) < 5:
        return 0.0, 0.0

    angles_np = np.array(angles, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    # robust: weighted median-ish via sorting
    idx = np.argsort(angles_np)
    angles_sorted = angles_np[idx]
    weights_sorted = weights_np[idx]
    cumw = np.cumsum(weights_sorted)
    cutoff = cumw[-1] * 0.5
    med_idx = int(np.searchsorted(cumw, cutoff))
    angle_med = float(angles_sorted[min(max(med_idx, 0), len(angles_sorted) - 1)])

    # confidence: concentration around median
    spread = float(np.average(np.abs(angles_np - angle_med), weights=weights_np))
    conf = max(0.0, min(1.0, 1.0 - (spread / 12.0)))  # 0..1

    # cap insane angles
    if abs(angle_med) > DESKEW_MAX_DEG:
        return 0.0, 0.0

    return angle_med, conf


def _rotate_bgr(bgr, angle_deg: float):
    import cv2
    import numpy as np

    if abs(angle_deg) < 0.01:
        return bgr

    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # compute new bounds to avoid cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        bgr, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _deskew_bgr(bgr) -> Tuple[Any, Optional[str]]:
    """
    Returns (deskewed_bgr, warning_message_or_None).
    """
    angle, conf = _estimate_skew_angle_deg(bgr)
    if conf < DESKEW_MIN_CONF or abs(angle) < 0.2:
        return bgr, None
    # To deskew, rotate by -angle
    deskewed = _rotate_bgr(bgr, -angle)
    return deskewed, f"deskew applied: angle={angle:.2f}deg conf={conf:.2f}"


# -----------------------------------------------------------------------------
# Table/box detection (medicine table crop)
# -----------------------------------------------------------------------------

def _find_largest_table_rect(bgr) -> Optional[Tuple[int, int, int, int]]:
    import cv2
    import numpy as np

    H, W = bgr.shape[:2]
    img_area = float(H * W)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    hor_k = max(15, W // 30)
    ver_k = max(15, H // 30)
    horizontal = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_k, 1)))
    vertical = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_k)))

    grid = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h)

        if area < img_area * MIN_TABLE_AREA_RATIO:
            continue
        if area > img_area * MAX_TABLE_AREA_RATIO:
            continue

        ar = w / float(h + 1e-6)
        if ar < 0.4 or ar > 4.5:
            continue

        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if not best:
        return None

    x, y, w, h = best
    x = max(0, x - TABLE_BORDER_PAD)
    y = max(0, y - TABLE_BORDER_PAD)
    w = min(W - x, w + 2 * TABLE_BORDER_PAD)
    h = min(H - y, h + 2 * TABLE_BORDER_PAD)
    return (x, y, w, h)


def _crop_bgr(bgr, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return bgr[y:y + h, x:x + w].copy()


# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------

def _ocr_bgr_blocking(bgr) -> str:
    import pytesseract

    processed = preprocess_for_ocr(bgr)
    pil_proc = _np_to_pil_gray(processed)

    # psm 6: block of text
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(pil_proc, lang=OCR_LANG, config=config)
    return text.strip()


def _ocr_pil_blocking(pil_img: Image.Image) -> str:
    pil_img = _pil_to_internal_jpeg(pil_img)
    bgr = _pil_to_bgr(pil_img)
    return _ocr_bgr_blocking(bgr)


def _ocr_with_focused_table_blocking(pil_img: Image.Image) -> Tuple[str, str, List[str]]:
    """
    Returns: (full_text, table_text, debug_warnings)
    Strategy:
    - Deskew whole image
    - Full OCR (fallback)
    - Detect table rect & OCR table crop (better for medicines)
    """
    warnings: List[str] = []

    # Make bgr
    pil_img = _pil_to_internal_jpeg(pil_img)
    bgr = _pil_to_bgr(pil_img)

    # Deskew
    bgr2, msg = _deskew_bgr(bgr)
    if msg:
        warnings.append(msg)

    # Full OCR
    full_text = _ocr_bgr_blocking(bgr2)

    # Table OCR
    rect = _find_largest_table_rect(bgr2)
    if not rect:
        return full_text, "", warnings

    table_bgr = _crop_bgr(bgr2, rect)
    table_text = _ocr_bgr_blocking(table_bgr)

    return full_text, table_text, warnings


# -----------------------------------------------------------------------------
# Parsing (medicines + dosing)
# -----------------------------------------------------------------------------

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


def _merge_text_for_parsing(full_text: str, table_text: str) -> str:
    full_text = (full_text or "").strip()
    table_text = (table_text or "").strip()
    if not table_text:
        return full_text
    return (full_text + "\n\n===TABLE_REGION===\n" + table_text).strip()


def _parse_prescription_blocking(text: str) -> Tuple[Dict[str, Optional[str]], List[Medicine], List[str]]:
    warnings: List[str] = []
    t = _clean_text(text)

    meta: Dict[str, Optional[str]] = {
        "patient_name": _find_first(
            [r"(?:환자명|성명|이름)\s*[:：]?\s*([^\n]+)", r"(?:Patient)\s*[:：]?\s*([^\n]+)"],
            t,
        ),
        "hospital": _find_first(
            [r"(?:병원명|의료기관명|요양기관명)\s*[:：]?\s*([^\n]+)", r"(?:의원|병원)\s*[:：]?\s*([^\n]+)"],
            t,
        ),
        "doctor": _find_first(
            [r"(?:의사명|처방의|담당의)\s*[:：]?\s*([^\n]+)", r"(?:Doctor)\s*[:：]?\s*([^\n]+)"],
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

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    meds: List[Medicine] = []

    # ---- patterns ----
    dose_pat = re.compile(
        r"(?:(?:1회|회당)\s*)?(\d+(?:\.\d+)?)\s*(정|캡슐|포|병|스푼|mg|g|mcg|㎎|㎍|mL|ML|cc)",
        re.IGNORECASE,
    )
    half_pat = re.compile(r"(?:반\s*정|0\.5\s*정|1/2\s*정)", re.IGNORECASE)

    tpd_pat = re.compile(
        r"(?:하루|1일)\s*(\d+)\s*회|(\d+)\s*회\s*/\s*일|(\d+)\s*times?\s*/\s*day|(?:BID|TID|QID)\b",
        re.IGNORECASE,
    )

    days_pat = re.compile(r"(\d+)\s*(?:일분|일치|일|days?)", re.IGNORECASE)

    instr_pat = re.compile(
        r"(식전|식후|식사전|식사후|공복|취침전|취침|아침|점심|저녁|필요시|PRN|as needed|통증시|발열시|기침시)",
        re.IGNORECASE,
    )

    def normalize_times_per_day(line: str, current: Optional[int]) -> Optional[int]:
        m = re.search(r"\b(BID|TID|QID)\b", line, re.IGNORECASE)
        if m:
            d = {"BID": 2, "TID": 3, "QID": 4}
            return d.get(m.group(1).upper(), current)

        hits = 0
        for k in ["아침", "점심", "저녁"]:
            if k in line:
                hits += 1
        if hits >= 2 and (current is None or current < hits):
            return hits
        return current

    def extract_instructions(line: str) -> Optional[str]:
        found = instr_pat.findall(line)
        if not found:
            return None
        uniq: List[str] = []
        for f in found:
            f2 = f.strip()
            if f2 and f2 not in uniq:
                uniq.append(f2)
        return ", ".join(uniq) if uniq else None

    # detect header region (table)
    header_idx = -1
    for i, ln in enumerate(lines):
        if any(k in ln for k in ["약품명", "처방 의약품", "처방의약품", "의약품", "복용", "투약", "용법", "용량", "1회", "1일"]):
            header_idx = i
            break

    for i, ln in enumerate(lines):
        # skip meta lines
        if any(k in ln for k in ["요양기관", "성명", "환자명", "교부일", "발행일", "주소", "전화", "진료과", "서명", "보험", "주민", "생년", "처방전", "e-mail"]):
            continue

        is_candidate = False
        if header_idx != -1 and i > header_idx:
            is_candidate = True
        if dose_pat.search(ln) or days_pat.search(ln) or tpd_pat.search(ln) or any(k in ln for k in ["복용", "투약", "용법", "용량"]):
            is_candidate = True
        if not is_candidate:
            continue

        # medicine name extraction: remove dosage tokens etc
        name = ln
        name = re.sub(r"(?:복용법|용법|용량)\s*[:：]?\s*", "", name)
        name = re.sub(r"(?:하루|1일)\s*\d+\s*회", "", name)
        name = re.sub(r"\d+\s*회\s*/\s*일", "", name)
        name = re.sub(r"\b(BID|TID|QID)\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\d+\s*(?:일분|일치|일|days?)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"(?:(?:1회|회당)\s*)?\d+(?:\.\d+)?\s*(정|캡슐|포|병|스푼|mg|g|mcg|㎎|㎍|mL|ML|cc)", "", name, flags=re.IGNORECASE)
        # remove typical instruction words
        for tok in ["식전", "식후", "식사전", "식사후", "공복", "취침전", "취침", "아침", "점심", "저녁", "필요시", "PRN", "as needed", "통증시", "발열시", "기침시"]:
            name = name.replace(tok, " ")
        name = re.sub(r"[–—\-:：|]+", " ", name).strip()
        name = re.sub(r"\s{2,}", " ", name)

        if len(name) < 2:
            name = ln.strip()

        # ignore meaningless name
        if re.fullmatch(r"[\d\W_]+", name or ""):
            continue

        # dose
        dose = None
        dm = dose_pat.search(ln)
        if dm:
            dose = f"{dm.group(1)}{dm.group(2)}"
        elif half_pat.search(ln):
            dose = "0.5정"

        # times/day
        times_per_day = None
        tm = tpd_pat.search(ln)
        if tm:
            for g in tm.groups():
                if g and g.isdigit():
                    times_per_day = int(g)
                    break
            times_per_day = normalize_times_per_day(ln, times_per_day)
        else:
            times_per_day = normalize_times_per_day(ln, None)

        # days
        days = None
        dsm = days_pat.search(ln)
        if dsm:
            try:
                days = int(dsm.group(1))
            except:
                days = None

        instructions = extract_instructions(ln)

        meds.append(
            Medicine(
                name=name,
                dose=dose,
                times_per_day=times_per_day,
                days=days,
                instructions=instructions,
                raw=ln,
            )
        )

    if not meds:
        warnings.append("약/복용정보 후보 라인을 찾지 못했습니다. (이미지 흐림/표 영역 인식/처방전 포맷 영향)")

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

    if file.content_type and not (
        file.content_type.startswith("image/") or file.content_type == "application/octet-stream"
    ):
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

    # normalize/resize
    try:
        pil_img = await run_in_threadpool(_coerce_pil, pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocess failed: {e}")

    # OCR (full + table focus) with deskew
    try:
        full_text, table_text, ocr_debug = await run_in_threadpool(_ocr_with_focused_table_blocking, pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    merged_text = await run_in_threadpool(_merge_text_for_parsing, full_text, table_text)

    meta, meds, parse_warnings = await run_in_threadpool(_parse_prescription_blocking, merged_text)

    # warnings: keep minimal & useful
    warnings: List[str] = []
    #  디버그 경고는 너무 길 수 있어 필요하면 켜기
    # warnings.extend(ocr_debug)
    warnings.extend(parse_warnings)

    return PrescriptionExtractResponse(
        success=True,
        text=full_text,  # 전체 텍스트를 반환 (보기 좋음). 약/복용 파싱은 merged_text 기반
        patient_name=meta.get("patient_name"),
        hospital=meta.get("hospital"),
        doctor=meta.get("doctor"),
        issued_date=meta.get("issued_date"),
        medicines=meds,
        warnings=warnings,
    )
