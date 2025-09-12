import os
import sys
import io
import json
import shutil
import uuid
import time
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field

# Add parent directory to path so we can import vision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from darwin.client import Client
from darwin.cli_functions import authenticate
import darwin.importer as importer
from darwin.importer import get_importer

from vision.prompts import make_prompt
from vision.client import get_client
from vision.config import DEFAULT_TEMPERATURE, DEFAULT_MAX_INPUT_SIZE
from vision.io_utils import load_image_from_path_or_url, resize_for_model
from vision.modes import run_boxes
from vision.config import pick_model


def _load_font(size: int = 16):
    try:
        return ImageFont.truetype("Arial.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("NotoSansCJK-Regular.ttc", size=size)
        except Exception:
            return ImageFont.load_default()


def _convert_box_to_pixels(box: List[float], order: str, orig_size: Tuple[int, int], model_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    W, H = orig_size
    mW, mH = model_size
    if order == "xy":
        x0, y0, x1, y1 = box
    else:
        y0, x0, y1, x1 = box
    maxv = max(float(x0), float(y0), float(x1), float(y1))
    # normalized 0..1
    if maxv <= 1.0001:
        x0p = int(float(x0) * W); y0p = int(float(y0) * H)
        x1p = int(float(x1) * W); y1p = int(float(y1) * H)
    # normalized 0..1000
    elif maxv <= 1000.0 + 1e-6:
        x0p = int(float(x0) / 1000.0 * W); y0p = int(float(y0) / 1000.0 * H)
        x1p = int(float(x1) / 1000.0 * W); y1p = int(float(y1) / 1000.0 * H)
    # pixels in original space
    elif float(x1) <= W + 2 and float(y1) <= H + 2:
        x0p = int(float(x0)); y0p = int(float(y0))
        x1p = int(float(x1)); y1p = int(float(y1))
    # pixels in model space (need scaling up to original)
    elif float(x1) <= mW + 2 and float(y1) <= mH + 2:
        sx = W / float(mW); sy = H / float(mH)
        x0p = int(float(x0) * sx); y0p = int(float(y0) * sy)
        x1p = int(float(x1) * sx); y1p = int(float(y1) * sy)
    else:
        # Fallback: try clamping
        x0p = int(float(x0)); y0p = int(float(y0))
        x1p = int(float(x1)); y1p = int(float(y1))
    # Order and clamp
    x0p, x1p = sorted([max(0, min(W - 1, x0p)), max(0, min(W - 1, x1p))])
    y0p, y1p = sorted([max(0, min(H - 1, y0p)), max(0, min(H - 1, y1p))])
    return x0p, y0p, x1p, y1p


def _save_debug_previews(orig: Image.Image, boxes: List[dict], model_size: Tuple[int, int], out_prefix: str) -> None:
    img = orig.convert("RGBA").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(16)
    for i, b in enumerate(boxes):
        x0p, y0p, x1p, y1p = _convert_box_to_pixels(b["box_2d"], "yx", orig.size, model_size)
        # Make outline more transparent
        draw.rectangle([(x0p, y0p), (x1p, y1p)], outline=(80, 220, 120, 180), width=3)
        # Make fill much more transparent (reduced alpha from 64 to 20)
        draw.rectangle([(x0p, y0p), (x1p, y1p)], fill=(80, 220, 120, 20))
        label = b.get("label", "shark")
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 6
        bx0, by0 = x0p + 8, max(0, y0p - th - pad * 2 - 8)
        # Make label background more transparent
        draw.rounded_rectangle([(bx0, by0), (bx0 + tw + pad * 2, by0 + th + pad * 2)], radius=8, fill=(60, 180, 80, 150))
        draw.text((bx0 + pad, by0 + pad), label, fill=(255, 255, 255, 255), font=font)
    out_path = f"{out_prefix}_preview.jpg"
    img.convert("RGB").save(out_path, quality=90)
    print(f"Saved debug preview: {out_prefix}_preview.jpg")


def format_v7_annotations(image_path: str, boxes: List[dict], class_names: List[str], orig_size: Tuple[int, int], model_size: Tuple[int, int]) -> dict:
    image_name = os.path.basename(image_path)
    annotations = []
    for b in boxes:
        # Our internal representation is [y0, x0, y1, x1]
        x0p, y0p, x1p, y1p = _convert_box_to_pixels(b["box_2d"], "yx", orig_size, model_size)
        label = b.get("label", "")
        if any(class_name.lower() in label.lower() for class_name in class_names):
            w = float(x1p - x0p)
            h = float(y1p - y0p)
            matched_class = next((c for c in class_names if c.lower() in label.lower()), class_names[0])
            annotations.append({
                "bounding_box": {"x": float(x0p), "y": float(y0p), "w": w, "h": h},
                "id": str(uuid.uuid4()),
                "name": matched_class,
                "slot_names": ["0"],
            })
    return {
        "version": "2.0",
        "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
        "item": {"name": image_name, "path": "/"},
        "annotations": annotations,
    }


def collect_images(path: str) -> List[str]:
    if os.path.isdir(path):
        supported = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(supported)]
    return [path]


@dataclass
class BBoxAutoConfig:
    # Inputs
    path: str = "../../data/example_images/paint_image1.jpg"
    items: str = "paint"
    prompt_prefix: Optional[str] = None
    class_names: List[str] = field(default_factory=lambda: ["paint"])  # filter labels
    # Model
    model: str = "gemini-2.5-pro"
    temperature: float = DEFAULT_TEMPERATURE
    max_input_size: int = DEFAULT_MAX_INPUT_SIZE
    # Filtering
    min_box: int = 24  # in normalized 0..1000 units (consistent with existing logic)
    # Behavior
    debug: bool = False
    local_only: bool = False
    v7_team: str = "clearobject"
    v7_dataset: str = "human-in-office-detection"


def run(cfg: BBoxAutoConfig) -> int:
    images = collect_images(cfg.path)
    if not images:
        print("No images found.")
        return 0

    # Prepare temp workspace
    base_dir = "./v7_upload_gemini"
    ann_dir = os.path.join(base_dir, "annotations")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(ann_dir)

    # Inference per image using vision package
    model = cfg.model or pick_model("boxes")
    client = get_client()  # ensure GOOGLE_API_KEY is set

    json_paths: List[str] = []
    uploaded_image_paths: List[str] = []

    for img_path in images:
        try:
            print(f"Processing {img_path}...")
            orig = load_image_from_path_or_url(img_path)
            model_img = resize_for_model(orig, cfg.max_input_size)
            prompt = make_prompt("boxes", cfg.items, None, 25, prefix=cfg.prompt_prefix)
            
            print(f"Prompt: {prompt}")
            print(f"Model: {model}")
            print(f"Image size: {orig.size} -> {model_img.size}")
            
            out = run_boxes(orig, model_img, prompt, model, cfg.temperature)
            
            raw_text = out.get('raw_text', 'No raw text')
            print(f"Raw response length: {len(raw_text)}")
            print(f"Raw response (first 1000 chars): {raw_text[:1000]}")
            
            boxes = out.get("boxes", [])
            print(f"Found {len(boxes)} boxes: {boxes}")
            
            # Filter tiny boxes if desired
            filtered = []
            for b in boxes:
                y0, x0, y1, x1 = b["box_2d"]
                if (y1 - y0) >= cfg.min_box and (x1 - x0) >= cfg.min_box:
                    filtered.append(b)
            
            if not filtered:
                print(f"No boxes for {img_path}")
                continue
                
            # Build V7 JSON with class filtering
            v7 = format_v7_annotations(img_path, filtered, cfg.class_names, orig.size, model_img.size)
            if not v7["annotations"]:
                print(f"No matching classes found for {img_path}")
                continue
            jpath = os.path.join(ann_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.json")
            with open(jpath, "w") as f:
                json.dump(v7, f, indent=2)
            json_paths.append(jpath)
            uploaded_image_paths.append(img_path)
            
            # Copy image to annotations folder if local-only mode
            if cfg.local_only:
                img_filename = os.path.basename(img_path)
                img_dest = os.path.join(ann_dir, img_filename)
                shutil.copy2(img_path, img_dest)
                print(f"Copied image to: {img_dest}")

            # Save local debug previews with different coordinate interpretations
            out_prefix = os.path.join(ann_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_preview")
            _save_debug_previews(orig, filtered, model_img.size, out_prefix)

            print(f"Annotated {img_path} with {len(v7['annotations'])} boxes")
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

    if not json_paths:
        print("No annotations generated; stopping.")
        shutil.rmtree(base_dir)
        return 0

    # Skip V7 upload if local-only mode
    if cfg.local_only:
        print(f"Local-only mode: Skipping V7 upload. Annotations and images saved to: {ann_dir}")
        print(f"Generated {len(json_paths)} annotation files.")
        return 0

    # Upload to V7
    load_dotenv()
    api_key = os.getenv("V7_API_KEY")
    if not api_key:
        print("V7_API_KEY not found. Create a .env with your key.")
        shutil.rmtree(base_dir)
        return

    dataset_slug = f"{cfg.v7_team}/{cfg.v7_dataset}"
    authenticate(api_key=api_key, default_team=True)
    client = Client.local()
    dataset = client.get_remote_dataset(dataset_slug)

    print(f"Uploading {len(uploaded_image_paths)} images to {dataset_slug}…")
    handler = dataset.push(uploaded_image_paths, multi_threaded=True)
    blocked = {item.filename for item in handler.blocked_items}
    errors = {os.path.basename(e.file_path) for e in handler.errors}
    failed = blocked.union(errors)

    json_to_import: List[str] = []
    for img, j in zip(uploaded_image_paths, json_paths):
        if os.path.basename(img) not in failed:
            json_to_import.append(j)

    if blocked:
        print(f"Skipped {len(blocked)} files (likely already exist).")
    if errors:
        print(f"Failed to upload {len(errors)} files due to errors.")

    if json_to_import:
        print(f"Importing {len(json_to_import)} annotation files to {dataset_slug}…")
        parser = get_importer("darwin")
        # Retry loop to wait for V7 to finish processing uploaded images
        max_attempts = 8
        delay = 3
        for attempt in range(1, max_attempts + 1):
            try:
                importer.import_annotations(dataset, parser, json_to_import, append=True, use_multi_cpu=True)
                print("Successfully imported annotations to V7.")
                break
            except ValueError as e:
                msg = str(e).lower()
                if ("processing" in msg or "failed to process" in msg) and attempt < max_attempts:
                    print(f"Files still processing on V7 (attempt {attempt}/{max_attempts}). Retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
    else:
        print("No new images were uploaded successfully; nothing to import.")

    shutil.rmtree(base_dir)


if __name__ == "__main__":
    # Edit the CONFIG values below to control behavior instead of passing CLI args.
    CONFIG = BBoxAutoConfig(
        path="../../data/example_images/paint_image1.jpg",
        items="paint",
        prompt_prefix=None,
        class_names=["paint"],
        model="gemini-2.5-pro",
        temperature=DEFAULT_TEMPERATURE,
        max_input_size=DEFAULT_MAX_INPUT_SIZE,
        min_box=24,
        debug=False,
        local_only=False,
        v7_team="clearobject",
        v7_dataset="human-in-office-detection",
    )
    run(CONFIG)
