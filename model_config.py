import os


MODEL_OPTIONS = (
    ("gemini-3.8-flash", "Gemini 3.8 Flash"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
)

MODEL_IDS = frozenset(model_id for model_id, _ in MODEL_OPTIONS)
LEGACY_MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-3.5-flash",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-1.5-flash": "gemini-3.5-flash",
    "gemini-1.5-pro": "gemini-3.5-flash",
    "gemini-3.8-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.5-flash-latest": "gemini-3.5-flash",
}

DEFAULT_MODEL = os.getenv("GOOGLE_MODEL") or MODEL_OPTIONS[0][0]
DEFAULT_MODEL = LEGACY_MODEL_ALIASES.get(DEFAULT_MODEL, DEFAULT_MODEL)
if DEFAULT_MODEL not in MODEL_IDS:
    DEFAULT_MODEL = MODEL_OPTIONS[0][0]
