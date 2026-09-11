"""Hybrid ANPR: EasyOCR on preprocessed crops + Moondream VLM on raw crops."""

import time

import config
from detectors.base_detector import BaseDetector
from detectors.moondream_reader import get_model, read_plate_vlm
from detectors.plate_localizer import detect_and_crop_plate
from detectors.plate_ocr import get_reader, read_plate
from detectors.plate_voter import PlateVoter
from engines.alert_engine import AlertType


class AnprDetector(BaseDetector):
    """Localizes plates, reads with EasyOCR + Moondream, confirms via consensus voting."""

    def __init__(
        self,
        gpu=None,
        use_moondream=None,
        alert_cooldown=None,
        min_plate_length=None,
        max_plate_length=None,
        voter_window=None,
        voter_min_votes=None,
    ):
        super().__init__()
        self.gpu = gpu if gpu is not None else config.EASYOCR_GPU
        self.use_moondream = use_moondream if use_moondream is not None else config.MOONDREAM_ENABLED
        self.alert_cooldown = alert_cooldown if alert_cooldown is not None else config.ANPR_ALERT_COOLDOWN
        self.min_plate_length = (
            min_plate_length if min_plate_length is not None else config.ANPR_MIN_PLATE_LENGTH
        )
        self.max_plate_length = (
            max_plate_length if max_plate_length is not None else config.ANPR_MAX_PLATE_LENGTH
        )

        self._reader = None
        self._vlm = None
        self._voter = PlateVoter(
            window=voter_window if voter_window is not None else config.ANPR_VOTER_WINDOW,
            min_votes=voter_min_votes if voter_min_votes is not None else config.ANPR_VOTER_MIN_VOTES,
        )
        self._last_alert_at = {}

    def on_start(self):
        self.log(f"loading EasyOCR (gpu={self.gpu})...")
        self._reader = get_reader(gpu=self.gpu)
        self.log("EasyOCR loaded")

        if self.use_moondream:
            self._vlm = get_model(log=self.log)

    def process(self, frame):
        plate_crop = detect_and_crop_plate(frame, self._reader)
        if plate_crop is None:
            return

        ocr_text = read_plate(
            plate_crop,
            self._reader,
            min_length=self.min_plate_length,
            max_length=self.max_plate_length,
        )
        vlm_text = None
        if self.use_moondream and self._vlm is not None:
            vlm_text = read_plate_vlm(
                plate_crop,
                model=self._vlm,
                min_length=self.min_plate_length,
                max_length=self.max_plate_length,
            )

        self._add_reads(ocr_text, vlm_text)

        confirmed = self._voter.get_consensus()
        if not confirmed:
            return

        if not self._cooldown_passed(confirmed):
            return

        sources = []
        if ocr_text:
            sources.append(f"ocr={ocr_text}")
        if vlm_text:
            sources.append(f"vlm={vlm_text}")

        self.alert(
            self.to_base64(frame),
            AlertType.NORMAL,
            f"plate detected: {confirmed} ({', '.join(sources)})",
        )

    def _add_reads(self, ocr_text, vlm_text):
        if ocr_text and vlm_text and ocr_text == vlm_text:
            self._voter.add(ocr_text)
            return

        if ocr_text:
            self._voter.add(ocr_text)
        if vlm_text:
            self._voter.add(vlm_text)

    def _cooldown_passed(self, plate):
        now = time.time()
        if now - self._last_alert_at.get(plate, 0) < self.alert_cooldown:
            return False
        self._last_alert_at[plate] = now
        return True
