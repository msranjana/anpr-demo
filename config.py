"""All settings are read from the .env file so nothing is hardcoded in the code."""

import os

from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/events.log")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))

# Moondream2 0.5B int4 ONNX bundle (auto-downloaded from HuggingFace when missing)
MOONDREAM_MODEL_PATH = os.getenv(
    "MOONDREAM_MODEL_PATH", "models/moondream/moondream-0_5b-int4.mf.gz"
)
MOONDREAM_REPO_ID = os.getenv("MOONDREAM_REPO_ID", "vikhyatk/moondream2")
MOONDREAM_MODEL_FILE = os.getenv("MOONDREAM_MODEL_FILE", "moondream-0_5b-int4.mf.gz")
MOONDREAM_REPO_REVISION = os.getenv(
    "MOONDREAM_REPO_REVISION", "9dddae84d54db4ac56fe37817aeaeb502ed083e2"
)

ANPR_PROMPT = os.getenv(
    "ANPR_PROMPT",
    "What is the vehicle license plate number visible in this image? "
    "Reply with only the plate number, or 'none' if no plate is visible.",
)
ANPR_FPS = int(os.getenv("ANPR_FPS", "1"))

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "")

# Comma separated list in .env -> list of addresses here
ALERT_TO_EMAILS = [
    email.strip() for email in os.getenv("ALERT_TO_EMAILS", "").split(",") if email.strip()
]
