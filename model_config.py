MODEL_OPTIONS = (
    ("qwen2.5:0.5b", "Qwen 2.5 0.5B (Ollama)"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash (Free tier)"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite (Free tier)"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"),
)

MODEL_IDS = frozenset(model_id for model_id, _ in MODEL_OPTIONS)
LEGACY_MODEL_ALIASES = {}
DEFAULT_MODEL = MODEL_OPTIONS[0][0]
