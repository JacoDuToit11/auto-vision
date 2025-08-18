import io
import os
import uuid
import base64
import requests
from typing import Tuple
from PIL import Image
from datetime import datetime



def load_image_from_path_or_url(path_or_url: str) -> Image.Image:
	if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
		r = requests.get(path_or_url, timeout=60)
		r.raise_for_status()
		data = io.BytesIO(r.content)
	else:
		data = io.BytesIO(open(path_or_url, "rb").read())
	im = Image.open(data)
	return im


def resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
	im = image.copy()
	w, h = im.size
	if max(w, h) <= max_side:
		return im
	scale = max_side / max(w, h)
	new_w = int(round(w * scale))
	new_h = int(round(h * scale))
	return im.resize((new_w, new_h), Image.Resampling.LANCZOS)


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def new_run_id() -> str:
	ts = datetime.now().strftime("%Y%m%d-%H%M%S")
	return f"{ts}_{uuid.uuid4()}"


def decode_data_uri_png(mask_data_uri: str) -> Image.Image:
	prefix = "data:image/png;base64,"
	if mask_data_uri.startswith(prefix):
		b64 = mask_data_uri[len(prefix):]
	else:
		b64 = mask_data_uri.split(",")[-1]
	raw = base64.b64decode(b64)
	return Image.open(io.BytesIO(raw))
