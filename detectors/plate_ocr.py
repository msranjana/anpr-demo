"""EasyOCR-based plate reading on cropped plate regions."""

import re

import cv2

_reader = None


def get_reader(gpu=False):
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=gpu)
    return _reader


def preprocess_plate(crop):
    """Upscale + denoise + threshold for better OCR on small/blurry plates."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    return gray


def clean_plate_text(text, min_length=4, max_length=12):
    cleaned = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if min_length <= len(cleaned) <= max_length:
        return cleaned
    return None


def read_plate(plate_crop, reader, min_length=4, max_length=12):
    """Run OCR on a cropped plate image, return cleaned plate string or None."""
    if plate_crop is None or plate_crop.size == 0:
        return None

    processed = preprocess_plate(plate_crop)
    results = reader.readtext(processed, detail=1)

    if not results:
        return None

    text = "".join([result[1] for result in results])
    return clean_plate_text(text, min_length=min_length, max_length=max_length)
