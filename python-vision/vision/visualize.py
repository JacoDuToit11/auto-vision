from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 16):
	try:
		return ImageFont.truetype("Arial.ttf", size=size)
	except Exception:
		try:
			return ImageFont.truetype("NotoSansCJK-Regular.ttc", size=size)
		except Exception:
			return ImageFont.load_default()


def _convert_box_to_pixels(box: List[float], img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
	"""Convert a box that might be [y0,x0,y1,x1] or [x0,y0,x1,y1] and might be
	normalized (0..1 or 0..1000) or in pixels into pixel coordinates.
	Returns (x0, y0, x1, y1) clamped and ordered.
	"""
	W, H = img_size
	def conv(vals: List[float], order: str) -> Tuple[int, int, int, int]:
		if order == "yx":
			y0, x0, y1, x1 = vals
		else:
			x0, y0, x1, y1 = vals
		mx = max(float(x0), float(x1)); my = max(float(y0), float(y1))
		# scale inference
		if max(mx, my) <= 1.0001: # 0..1
			x0p = int(float(x0) * W); y0p = int(float(y0) * H)
			x1p = int(float(x1) * W); y1p = int(float(y1) * H)
		elif max(mx, my) <= 1000.0 + 1e-6: # 0..1000
			x0p = int(float(x0) / 1000.0 * W); y0p = int(float(y0) / 1000.0 * H)
			x1p = int(float(x1) / 1000.0 * W); y1p = int(float(y1) / 1000.0 * H)
		else: # assume pixels
			x0p = int(float(x0)); y0p = int(float(y0))
			x1p = int(float(x1)); y1p = int(float(y1))
		# clamp and order
		x0p, x1p = sorted([max(0, min(W - 1, x0p)), max(0, min(W - 1, x1p))])
		y0p, y1p = sorted([max(0, min(H - 1, y0p)), max(0, min(H - 1, y1p))])
		return x0p, y0p, x1p, y1p
	
	# Evaluate both orders and pick the more plausible (area within image)
	cand_yx = conv(box, "yx")
	cand_xy = conv(box, "xy")
	def score(c: Tuple[int,int,int,int]) -> float:
		x0,y0,x1,y1 = c
		w = max(0, x1 - x0); h = max(0, y1 - y0)
		if w == 0 or h == 0: return -1
		# prefer boxes that are not almost the whole image
		area = w * h
		return -abs((area / (W * H)) - 0.25)  # prefer around 25% as a weak prior
	return cand_yx if score(cand_yx) >= score(cand_xy) else cand_xy


def draw_boxes(img: Image.Image, boxes: List[Dict]) -> Image.Image:
	drawn = img.convert("RGBA").copy()
	draw = ImageDraw.Draw(drawn)
	width, height = drawn.size
	colors = ["#ef4444","#22c55e","#3b82f6","#eab308","#a855f7","#f97316","#06b6d4","#84cc16"]
	for i, b in enumerate(boxes):
		color = colors[i % len(colors)]
		x0p, y0p, x1p, y1p = _convert_box_to_pixels(b["box_2d"], (width, height))
		draw.rectangle([(x0p,y0p),(x1p,y1p)], outline=color, width=4)
		# translucent fill
		fill = tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (70,)
		draw.rectangle([(x0p,y0p),(x1p,y1p)], fill=fill)
	return drawn.convert("RGB")


def draw_points(img: Image.Image, points: List[Dict]) -> Image.Image:
	drawn = img.convert("RGBA").copy()
	draw = ImageDraw.Draw(drawn)
	width, height = drawn.size
	colors = ["#ef4444","#22c55e","#3b82f6","#eab308","#a855f7","#f97316","#06b6d4","#84cc16"]
	for i, p in enumerate(points):
		color = colors[i % len(colors)]
		# Points may be [y,x] normalized/pixels; reuse box logic by treating as a tiny box
		x0p, y0p, x1p, y1p = _convert_box_to_pixels([p["point"][0], p["point"][1], p["point"][0], p["point"][1]], (width, height))
		xc, yc = x0p, y0p
		r = 6
		draw.ellipse([(xc-r,yc-r),(xc+r,yc+r)], fill=color, outline="#000000")
	return drawn.convert("RGB")


def overlay_masks(img: Image.Image, entries: List[Dict]) -> Image.Image:
	base = img.convert("RGBA").copy()
	width, height = base.size
	for i, e in enumerate(entries):
		mask = e["mask_image"].convert("L").resize((width, height))
		color = (255, 0, 0, 90)
		overlay = Image.new("RGBA", (width, height), color)
		base = Image.composite(overlay, base, mask)
	return base.convert("RGB")
