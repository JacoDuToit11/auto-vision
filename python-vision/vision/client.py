import os
from google import genai


def get_client():
	api_key = os.getenv("GOOGLE_API_KEY")
	if not api_key:
		raise RuntimeError("GOOGLE_API_KEY not set")
	return genai.Client(api_key=api_key)
