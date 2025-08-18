import json
import time
from typing import Any, Dict, List
from PIL import Image

from google.genai import types
from google.genai.errors import ServerError

from .client import get_client
from .config import SAFETY_SETTINGS
from .io_utils import decode_data_uri_png
from .visualize import draw_boxes, draw_points, overlay_masks


def _extract_top_level_array(text: str) -> str:
	if "```json" in text:
		text = text.split("```json",1)[1].split("```",1)[0]
	elif "```" in text:
		text = text.split("```",1)[1].split("```",1)[0]
	start = text.find("[")
	if start == -1:
		return text.strip()
	depth = 0
	for i,ch in enumerate(text[start:], start):
		if ch == "[": depth += 1
		elif ch == "]":
			depth -= 1
			if depth == 0:
				return text[start:i+1]
	return text[start:].strip()


def _generate_with_image(prompt: str, image: Image.Image, model: str, temperature: float) -> str:
	client = get_client()
	max_retries = 3
	retry_delay = 1  # seconds
	
	for attempt in range(max_retries):
		try:
			response = client.models.generate_content(
				model=model,
				contents=[prompt, image],
				config=types.GenerateContentConfig(
					temperature=temperature,
					safety_settings=SAFETY_SETTINGS,
				),
			)
			result = response.text or ""
			
			# If we got a response, return it
			if result.strip():
				return result
			
			# Empty response, retry if we have attempts left
			if attempt < max_retries - 1:
				print(f"Empty response from model, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
				time.sleep(retry_delay)
				retry_delay *= 2  # Exponential backoff
				continue
			else:
				print(f"Empty response after {max_retries} attempts")
				return ""
				
		except ServerError as e:
			if attempt < max_retries - 1:
				print(f"Server error (attempt {attempt + 1}/{max_retries}): {e}")
				print(f"Retrying in {retry_delay}s...")
				time.sleep(retry_delay)
				retry_delay *= 2  # Exponential backoff
				continue
			else:
				print(f"Server error after {max_retries} attempts: {e}")
				return ""
		except Exception as e:
			print(f"Unexpected error: {e}")
			return ""
	
	return ""


def run_boxes(orig_image: Image.Image, model_image: Image.Image, prompt: str, model: str, temperature: float) -> Dict[str, Any]:
	raw = _generate_with_image(prompt, model_image, model, temperature)
	json_text = _extract_top_level_array(raw)
	
	# Handle empty or invalid JSON gracefully
	if not json_text.strip():
		print("No response from model")
		return {"boxes": [], "image": orig_image, "raw_text": raw}
	
	try:
		items = json.loads(json_text)
	except json.JSONDecodeError as e:
		print(f"Failed to parse JSON response: {e}")
		print(f"Raw text: {raw[:500]}...")
		return {"boxes": [], "image": orig_image, "raw_text": raw}
	
	boxes: List[Dict[str, Any]] = []
	for it in items:
		if not isinstance(it, dict):
			continue
		box = it.get("box_2d")
		if isinstance(box, list) and len(box) == 4:
			boxes.append({"label": it.get("label"), "box_2d": [int(v) for v in box]})
	vis = draw_boxes(orig_image, boxes)
	return {"boxes": boxes, "image": vis, "raw_text": raw}


def run_points(orig_image: Image.Image, model_image: Image.Image, prompt: str, model: str, temperature: float) -> Dict[str, Any]:
	raw = _generate_with_image(prompt, model_image, model, temperature)
	json_text = _extract_top_level_array(raw)
	
	if not json_text.strip():
		print("No response from model")
		return {"points": [], "image": orig_image, "raw_text": raw}
	
	try:
		items = json.loads(json_text)
	except json.JSONDecodeError as e:
		print(f"Failed to parse JSON response: {e}")
		return {"points": [], "image": orig_image, "raw_text": raw}
	
	points: List[Dict[str, Any]] = []
	for it in items:
		if not isinstance(it, dict):
			continue
		pt = it.get("point")
		if isinstance(pt, list) and len(pt) == 2:
			points.append({"point": [float(pt[0]), float(pt[1])], "label": it.get("label")})
	vis = draw_points(orig_image, points)
	return {"points": points, "image": vis, "raw_text": raw}


def run_seg(orig_image: Image.Image, model_image: Image.Image, prompt: str, model: str, temperature: float) -> Dict[str, Any]:
	raw = _generate_with_image(prompt, model_image, model, temperature)
	json_text = _extract_top_level_array(raw)
	
	if not json_text.strip():
		print("No response from model")
		return {"masks": [], "image": orig_image, "raw_text": raw}
	
	try:
		items = json.loads(json_text)
	except json.JSONDecodeError as e:
		print(f"Failed to parse JSON response: {e}")
		return {"masks": [], "image": orig_image, "raw_text": raw}
	
	entries: List[Dict[str, Any]] = []
	for it in items:
		if not isinstance(it, dict):
			continue
		mask_uri = it.get("mask")
		box = it.get("box_2d")
		if mask_uri and isinstance(box, list) and len(box) == 4:
			try:
				mask_im = decode_data_uri_png(mask_uri)
				entries.append({"label": it.get("label"), "box_2d": [int(v) for v in box], "mask_image": mask_im})
			except Exception:
				continue
	vis = overlay_masks(orig_image, entries)
	return {"masks": [{"label": e["label"], "box_2d": e["box_2d"]} for e in entries], "image": vis, "raw_text": raw}


def run_boxes3d(orig_image: Image.Image, model_image: Image.Image, prompt: str, model: str, temperature: float) -> Dict[str, Any]:
	raw = _generate_with_image(prompt, model_image, model, temperature)
	json_text = _extract_top_level_array(raw)
	
	if not json_text.strip():
		print("No response from model")
		return {"boxes3d": [], "raw_text": raw}
	
	try:
		items = json.loads(json_text)
	except json.JSONDecodeError as e:
		print(f"Failed to parse JSON response: {e}")
		return {"boxes3d": [], "raw_text": raw}
	
	boxes3d: List[Dict[str, Any]] = []
	for it in items:
		if not isinstance(it, dict):
			continue
		b = it.get("box_3d")
		if isinstance(b, list) and len(b) == 9:
			boxes3d.append({"label": it.get("label"), "box_3d": [float(v) for v in b]})
	return {"boxes3d": boxes3d, "raw_text": raw}
