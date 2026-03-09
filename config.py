# ai_layer/config.py
# All models are from openrouter

# ⚙️ Main Bot Model
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"
DEFAULT_TEMPERATURE = 0.2

# ✨ A list of models to try if the default one fails with a 503 error.
FALLBACK_MODELS = [
    "anthropic/claude-3.5-sonnet:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "google/gemini-pro-1.5:free"
]