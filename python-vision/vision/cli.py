import os
import json
import argparse
from typing import Any, Dict
from PIL import Image

from .config import DEFAULT_TEMPERATURE, DEFAULT_MAX_ITEMS, pick_model
from .io_utils import load_image_from_path_or_url, ensure_dir, new_run_id, resize_for_model
from .prompts import make_prompt
from .modes import run_boxes, run_seg, run_points, run_boxes3d


def main() -> int:
	parser = argparse.ArgumentParser(description="Python Vision CLI (TSX-parity)")
	parser.add_argument("--image", required=True, help="Path or URL to image")
	parser.add_argument("--mode", required=True, choices=["boxes", "seg", "points", "boxes3d"]) 
	parser.add_argument("--prompt", help="Optional prompt override")
	parser.add_argument("--items", help="Target items text (e.g., 'cars') for modes that support it")
	parser.add_argument("--seg-language", help="Language name for segmentation labels (e.g., Deutsch)")
	parser.add_argument("--model", help="Model name (defaults to 2.5 Flash, 3D uses 2.0 Flash)")
	parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
	parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
	parser.add_argument("--max-input-size", type=int, default=640, help="Resize longest side to this for model input")
	parser.add_argument("--out", default="outputs")
	parser.add_argument("--output-pixels", action="store_true", help="Include pixel-coord copies in JSON outputs")
	args = parser.parse_args()

	ensure_dir(args.out)
	run_id = new_run_id()

	orig_img = load_image_from_path_or_url(args.image)
	model_img = resize_for_model(orig_img, args.max_input_size)
	model = args.model or pick_model(args.mode)
	prompt = args.prompt or make_prompt(args.mode, args.items, args.seg_language, args.max_items)

	if args.mode == "boxes":
		out = run_boxes(orig_img, model_img, prompt, model, args.temperature)
		out_image = out["image"]
		result: Dict[str, Any] = {"mode":"boxes","boxes":out["boxes"],"prompt_used":prompt,"model_name":model}
		if args.output_pixels:
			W, H = orig_img.size
			result["boxes_px"] = [
				{
					"label": b.get("label"),
					"x0": int(b["box_2d"][1]/1000 * W),
					"y0": int(b["box_2d"][0]/1000 * H),
					"x1": int(b["box_2d"][3]/1000 * W),
					"y1": int(b["box_2d"][2]/1000 * H),
				}
				for b in out["boxes"]
			]
	elif args.mode == "seg":
		out = run_seg(orig_img, model_img, prompt, model, args.temperature)
		out_image = out["image"]
		result = {"mode":"seg","masks":out["masks"],"prompt_used":prompt,"model_name":model}
	elif args.mode == "points":
		out = run_points(orig_img, model_img, prompt, model, args.temperature)
		out_image = out["image"]
		result = {"mode":"points","points":out["points"],"prompt_used":prompt,"model_name":model}
		if args.output_pixels:
			W, H = orig_img.size
			result["points_px"] = [
				{
					"label": p.get("label"),
					"x": int(p["point"][1]/1000 * W),
					"y": int(p["point"][0]/1000 * H),
				}
				for p in out["points"]
			]
	else:
		out = run_boxes3d(orig_img, model_img, prompt, model, args.temperature)
		out_image = None
		result = {"mode":"boxes3d","boxes3d":out["boxes3d"],"prompt_used":prompt,"model_name":model}

	json_path = os.path.join(args.out, f"{run_id}.json")
	with open(json_path, "w") as f:
		json.dump(result, f, indent=2)

	if out_image is not None:
		image_path = os.path.join(args.out, f"{run_id}.jpg")
		out_image.save(image_path, format="JPEG", quality=90)
		print(f"Saved: {json_path}\nSaved: {image_path}")
	else:
		print(f"Saved: {json_path}\n(No 2D visualization for 3D boxes)")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
