import os
from google.genai import types

# Global defaults (override via env if needed)
MODELS = [
	"gemini-2.5-flash",
	"gemini-2.5-pro",
	"gemini-2.0-flash",
]
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", MODELS[0])
DEFAULT_MAX_ITEMS = 25
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_INPUT_SIZE = int(os.getenv("DEFAULT_MAX_INPUT_SIZE", "640"))


def pick_model(mode: str) -> str:
	# Prefer an explicit 2.0 flash for 3D if available; otherwise use DEFAULT_MODEL
	if mode == "boxes3d" and any(m.startswith("gemini-2.0-flash") for m in MODELS):
		return "gemini-2.0-flash"
	return DEFAULT_MODEL


SAFETY_SETTINGS = [
	types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]
