"""Automatic number plate recognition using Moondream2 ONNX (int4 .mf.gz bundle)."""

import os
import re
import time

import cv2
from PIL import Image

import config
from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType


class AnprDetector(BaseDetector):
    """Reads license plates from CCTV frames with the quantized Moondream2 0.5B model."""

    _PLATE_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9\s\-]{3,}[A-Z0-9]")
    _REJECT_PATTERN = re.compile(
        r"\b(none|unknown|not visible|no plate|n/?a|cannot|can't|unable)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model_path=None,
        prompt=None,
        max_new_tokens=32,
        min_plate_length=4,
        alert_cooldown=15,
    ):
        super().__init__()
        self.model_path = model_path or config.MOONDREAM_MODEL_PATH
        self.prompt = prompt or config.ANPR_PROMPT
        self.max_new_tokens = max_new_tokens
        self.min_plate_length = min_plate_length
        self.alert_cooldown = alert_cooldown

        self._model = None
        self._last_alert_at = {}

    def on_start(self):
        import moondream as md

        path = self.model_path
        if not os.path.isfile(path):
            self.log(f"downloading Moondream ONNX from {config.MOONDREAM_REPO_ID}...")
            path = self._resolve_model_path()

        self.log(f"loading Moondream ONNX from {path}...")
        self._model = md.vl(model=path)
        self.log("Moondream ONNX model loaded")

    def process(self, frame):
        answer = self._ask(frame)
        plate = self._parse_plate(answer)
        if not plate:
            return

        if not self._cooldown_passed(plate):
            return

        self.alert(
            self.to_base64(frame),
            AlertType.NORMAL,
            f"plate detected: {plate} (vlm: {answer})",
        )

    def _resolve_model_path(self):
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

    def _ask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        result = self._model.query(
            pil_image,
            self.prompt,
            settings={"max_tokens": self.max_new_tokens},
        )
        return (result.get("answer") or "").strip()

    def _parse_plate(self, answer):
        text = (answer or "").strip()
        if not text or self._REJECT_PATTERN.search(text):
            return None

        text = re.sub(
            r"^(license plate|plate number|number plate)\s*[:\-]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip(" .\"'")
        text = re.sub(r"\s+", " ", text).upper()

        match = self._PLATE_PATTERN.search(text)
        if not match:
            return None

        plate = re.sub(r"\s+", "", match.group(0))
        if len(plate) < self.min_plate_length:
            return None
        return plate

    def _cooldown_passed(self, plate):
        now = time.time()
        if now - self._last_alert_at.get(plate, 0) < self.alert_cooldown:
            return False
        self._last_alert_at[plate] = now
        return True
