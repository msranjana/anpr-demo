"""Moondream VLM plate reading on raw plate crops (no preprocessing)."""

import os

import cv2
from PIL import Image

import config
from detectors.plate_ocr import clean_plate_text

_model = None


def get_model(log=None):
    global _model
    if _model is not None:
        return _model

    import moondream as md

    path = config.MOONDREAM_MODEL_PATH
    if not os.path.isfile(path):
        if log:
            log(f"downloading Moondream from {config.MOONDREAM_REPO_ID}...")
        path = _resolve_model_path()

    if log:
        log(f"loading Moondream from {path}...")
    _model = md.vl(model=path)
    if log:
        log("Moondream model loaded")
    return _model


def _resolve_model_path():
    if config.MOONDREAM_MODEL_PATH and os.path.isfile(config.MOONDREAM_MODEL_PATH):
        return config.MOONDREAM_MODEL_PATH

    folder = os.path.dirname(config.MOONDREAM_MODEL_PATH) or "models/moondream"
    os.makedirs(folder, exist_ok=True)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=config.MOONDREAM_REPO_ID,
        filename=config.MOONDREAM_MODEL_FILE,
        revision=config.MOONDREAM_REPO_REVISION,
        local_dir=folder,
    )


def read_plate_vlm(plate_crop, model=None, min_length=4, max_length=12):
    """Read plate text from a raw crop using Moondream VLM OCR (no preprocessing)."""
    if plate_crop is None or plate_crop.size == 0:
        return None

    model = model or get_model()
    rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    result = model.query(
        pil_image,
        config.MOONDREAM_ANPR_PROMPT,
        settings={"max_tokens": config.MOONDREAM_MAX_TOKENS},
    )
    answer = (result.get("answer") or "").strip()
    return clean_plate_text(answer, min_length=min_length, max_length=max_length)
