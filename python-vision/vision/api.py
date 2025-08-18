import os
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import pick_model, DEFAULT_TEMPERATURE, DEFAULT_MAX_ITEMS, DEFAULT_MAX_INPUT_SIZE
from .io_utils import new_run_id, resize_for_model, ensure_dir
from .prompts import make_prompt
from .modes import run_boxes, run_seg, run_points, run_boxes3d

app = FastAPI(title="Python Vision API", version="0.1.1")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Ensure outputs directory exists and mount static files
OUTPUT_DIR = "outputs"
ensure_dir(OUTPUT_DIR)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/detect")
async def detect(
	file: UploadFile = File(...),
	mode: str = Form("boxes"),
	prompt: Optional[str] = Form(None),
	items: Optional[str] = Form(None),
	model: Optional[str] = Form(None),
	temperature: float = Form(DEFAULT_TEMPERATURE),
	max_items: int = Form(DEFAULT_MAX_ITEMS),
	max_input_size: int = Form(DEFAULT_MAX_INPUT_SIZE),
):
	if mode not in ("boxes","seg","points","boxes3d"):
		raise HTTPException(status_code=400, detail="invalid mode")
	data = await file.read()
	try:
		orig_img = Image.open(io.BytesIO(data))
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"invalid image: {e}")

	model_img = resize_for_model(orig_img, max_input_size) if max_input_size else orig_img
	model_name = model or pick_model(mode)
	final_prompt = prompt or make_prompt(mode, items, None, max_items)

	# Save results
	run_id = new_run_id()
	json_path = os.path.join(OUTPUT_DIR, f"{run_id}.json")
	image_path = os.path.join(OUTPUT_DIR, f"{run_id}.jpg")

	if mode == "boxes":
		out = run_boxes(orig_img, model_img, final_prompt, model_name, temperature)
		result = {"mode":"boxes","boxes":out["boxes"],"prompt_used":final_prompt,"model_name":model_name}
		if out["image"]:
			out["image"].save(image_path, format="JPEG", quality=90)
			result["image_url"] = f"/outputs/{run_id}.jpg"
	elif mode == "seg":
		out = run_seg(orig_img, model_img, final_prompt, model_name, temperature)
		result = {"mode":"seg","masks":out["masks"],"prompt_used":final_prompt,"model_name":model_name}
		if out["image"]:
			out["image"].save(image_path, format="JPEG", quality=90)
			result["image_url"] = f"/outputs/{run_id}.jpg"
	elif mode == "points":
		out = run_points(orig_img, model_img, final_prompt, model_name, temperature)
		result = {"mode":"points","points":out["points"],"prompt_used":final_prompt,"model_name":model_name}
		if out["image"]:
			out["image"].save(image_path, format="JPEG", quality=90)
			result["image_url"] = f"/outputs/{run_id}.jpg"
	else:
		out = run_boxes3d(orig_img, model_img, final_prompt, model_name, temperature)
		result = {"mode":"boxes3d","boxes3d":out["boxes3d"],"prompt_used":final_prompt,"model_name":model_name}

	# Save JSON
	import json
	with open(json_path, "w") as f:
		json.dump(result, f, indent=2)

	return JSONResponse(result)
