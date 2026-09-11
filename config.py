"""All settings are read from the .env file so nothing is hardcoded in the code."""

import os

from dotenv import load_dotenv

# Load Untitled (project env template); .env overrides if present.
load_dotenv("Untitled")
load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/events.log")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))

ANPR_FPS = int(os.getenv("ANPR_FPS", "2"))
EASYOCR_GPU = os.getenv("EASYOCR_GPU", "false").lower() in {"1", "true", "yes"}

# Moondream VLM reads raw plate crops without preprocessing (hybrid with EasyOCR)
MOONDREAM_ENABLED = os.getenv("MOONDREAM_ENABLED", "true").lower() in {"1", "true", "yes"}
MOONDREAM_MODEL_PATH = os.getenv(
    "MOONDREAM_MODEL_PATH", "models/moondream/moondream-0_5b-int4.mf.gz"
)
MOONDREAM_REPO_ID = os.getenv("MOONDREAM_REPO_ID", "vikhyatk/moondream2")
MOONDREAM_MODEL_FILE = os.getenv("MOONDREAM_MODEL_FILE", "moondream-0_5b-int4.mf.gz")
MOONDREAM_REPO_REVISION = os.getenv(
    "MOONDREAM_REPO_REVISION", "9dddae84d54db4ac56fe37817aeaeb502ed083e2"
)
MOONDREAM_ANPR_PROMPT = os.getenv(
    "MOONDREAM_ANPR_PROMPT",
    "Read the license plate characters in this image. "
    "Reply with only the plate number, or 'none' if unreadable.",
)
MOONDREAM_MAX_TOKENS = int(os.getenv("MOONDREAM_MAX_TOKENS", "32"))
ANPR_MIN_PLATE_LENGTH = int(os.getenv("ANPR_MIN_PLATE_LENGTH", "4"))
ANPR_MAX_PLATE_LENGTH = int(os.getenv("ANPR_MAX_PLATE_LENGTH", "12"))
ANPR_VOTER_WINDOW = int(os.getenv("ANPR_VOTER_WINDOW", "5"))
ANPR_VOTER_MIN_VOTES = int(os.getenv("ANPR_VOTER_MIN_VOTES", "3"))
ANPR_ALERT_COOLDOWN = int(os.getenv("ANPR_ALERT_COOLDOWN", "15"))

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "")

# Comma separated list in .env -> list of addresses here
ALERT_TO_EMAILS = [
    email.strip() for email in os.getenv("ALERT_TO_EMAILS", "").split(",") if email.strip()
]
