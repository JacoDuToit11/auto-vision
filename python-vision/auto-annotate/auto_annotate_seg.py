import os
import sys
import json
import shutil
import uuid
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageDraw
from dataclasses import dataclass, field

# Add parent directory to path so we can import vision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from darwin.client import Client
from darwin.cli_functions import authenticate
import darwin.importer as importer
from darwin.importer import get_importer

from vision.prompts import make_prompt
from vision.config import DEFAULT_TEMPERATURE, DEFAULT_MAX_INPUT_SIZE, pick_model
from vision.io_utils import load_image_from_path_or_url, resize_for_model, decode_data_uri_png
from vision.modes import run_seg
from vision.visualize import overlay_masks


def _extract_top_level_array(text: str) -> str:
    """Extract the first top-level JSON array substring from a possibly fenced or chatty response."""
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    # Try to find a complete JSON array first
    start = text.find("[")
    if start == -1:
        return text.strip()

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # If we can't find a complete array, try to extract a partial but valid JSON
    # Look for the last complete object in the array
    try:
        # Find all object starts and ends
        brace_depth = 0
        last_valid_end = start - 1

        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    last_valid_end = i

        if last_valid_end > start:
            # Try to construct a valid array with the complete objects we found
            partial_array = text[start:last_valid_end + 1]
            if partial_array.count("{") == partial_array.count("}"):
                return partial_array

    except:
        pass

    # As a last resort, return what we have
    return text[start:].strip()


def collect_images(path: str) -> List[str]:
    if os.path.isdir(path):
        supported = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(supported)]
    return [path]


def mask_to_polygons(binary_mask: np.ndarray,
                     min_area: int = 100,
                     approx_eps_frac: float = 0.01,
                     export_all: bool = False) -> List[np.ndarray]:
    """Convert a binary mask (uint8 0/255) to one or more simplified polygons.

    Returns a list of polygons, each shaped (N, 2) with x,y coordinates.
    """
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1.0, approx_eps_frac * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        if approx.shape[0] < 3:
            continue
        polys.append(approx.reshape(-1, 2))
    if not export_all and polys:
        # Keep only the largest polygon
        areas = [cv2.contourArea(p.reshape(-1, 1, 2)) for p in polys]
        idx = int(np.argmax(areas))
        return [polys[idx]]
    return polys


def _draw_polygons_outline(pil_img: Image.Image, polygons: List[np.ndarray], color: Tuple[int, int, int], width: int = 3) -> Image.Image:
    """Draw polygon outlines on a PIL image.
    Polygons are arrays of shape (N, 2) in absolute pixel coords.
    """
    img = pil_img.convert("RGBA").copy()
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = [(int(x), int(y)) for x, y in poly]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color + (255,), width=width)
    return img.convert("RGB")


def polygon_to_v7_path(poly: np.ndarray) -> List[dict]:
    path: List[dict] = []
    for x, y in poly:
        path.append({"x": float(x), "y": float(y)})
    return path


def format_v7_polygon_annotations(image_path: str,
                                  polygons_with_labels: List[Tuple[np.ndarray, str]]) -> dict:
    image_name = os.path.basename(image_path)
    annotations: List[dict] = []
    for poly, label in polygons_with_labels:
        annotations.append({
            "polygon": {"path": polygon_to_v7_path(poly)},
            "id": str(uuid.uuid4()),
            "name": label,
            "slot_names": ["0"],
        })
    return {
        "version": "2.0",
        "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
        "item": {"name": image_name, "path": "/"},
        "annotations": annotations,
    }


def _choose_class_name(predicted_label: Optional[str], class_names: List[str]) -> str:
    if predicted_label:
        for cname in class_names:
            if cname.lower() in predicted_label.lower():
                return cname
    return class_names[0] if class_names else (predicted_label or "object")


@dataclass
class SegAutoConfig:
    # Inputs
    path: str = "../../data/example_images/paint_image1.jpg"
    items: str = "ink"
    prompt_prefix: Optional[str] = "Give the segmentation mask for the ink deposit in the middle of the printing cylinder."
    prompt_override: Optional[str] = None  # If set, completely overrides the entire prompt
    class_names: List[str] = field(default_factory=lambda: ["ink"])  # filter/map labels
    # Model
    model: str = "gemini-2.5-flash"
    temperature: float = DEFAULT_TEMPERATURE
    max_input_size: int = DEFAULT_MAX_INPUT_SIZE
    # Segmentation mode
    mode: str = "single"  # "single" for one object, "multiple" for list of objects
    # How to interpret model masks when exporting JSON
    #   "box": mask is relative to the bbox (resize to box, then offset into image). Default.
    #   "full": mask is over the full image (resize to image dims directly).
    mask_interpretation: str = "box"
    # Number of preview images to save per input (1: box only, 2: box+full, 3: box+full+orig)
    preview_count: int = 3
    # Number of independent attempts/variants per image (different model calls)
    variants_per_image: int = 3
    # Polygon extraction
    min_area: int = 100
    approx_eps_frac: float = 0.01
    export_all: bool = False
    # Output / behavior
    debug: bool = False
    local_only: bool = True
    v7_team: str = "clearobject"
    v7_dataset: str = "meyers-rsp"


def run(cfg: SegAutoConfig) -> int:
    images = collect_images(cfg.path)
    if not images:
        print("No images found.")
        return 0

    # Prepare temp workspace
    base_dir = "./v7_upload_seg"
    ann_dir = os.path.join(base_dir, "annotations")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(ann_dir)

    model = cfg.model or pick_model("seg")

    json_paths: List[str] = []
    uploaded_image_paths: List[str] = []

    for img_path in images:
        try:
            print(f"Processing {img_path}...")
            orig = load_image_from_path_or_url(img_path)
            model_img = resize_for_model(orig, cfg.max_input_size)
            # Use prompt override if provided, otherwise generate from components
            if cfg.prompt_override:
                prompt = cfg.prompt_override
            else:
                prompt = make_prompt("seg", cfg.items, None, 25, prefix=cfg.prompt_prefix, seg_mode=cfg.mode)

            if cfg.debug:
                print(f"Prompt: {prompt}")
                print(f"Model: {model}")
                print(f"Image size: {orig.size} -> {model_img.size}")

            num_variants = max(1, int(cfg.variants_per_image))
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for variant_idx in range(num_variants):
                temp_to_use = max(0.0, min(2.0, cfg.temperature + (0.05 * (variant_idx - (num_variants-1)/2))))
                out = run_seg(orig, model_img, prompt, model, temp_to_use)

                raw_text = out.get("raw_text", "")
                if cfg.debug:
                    print(f"[v{variant_idx+1}/{num_variants}] Raw response length: {len(raw_text)}")
                    print(f"[v{variant_idx+1}/{num_variants}] Raw head: {raw_text[:200]}")

                if not raw_text.strip():
                    print(f"Model returned empty response for {img_path} (variant {variant_idx+1}/{num_variants})")
                    continue

                json_text = _extract_top_level_array(raw_text)

                # Try multiple approaches to parse the JSON
                items = []
                try:
                    if json_text.strip():
                        parsed = json.loads(json_text)
                        if cfg.mode == "single":
                            if isinstance(parsed, dict):
                                items = [parsed]
                            else:
                                items = parsed if isinstance(parsed, list) else [parsed]
                        else:
                            items = parsed if isinstance(parsed, list) else [parsed]
                except json.JSONDecodeError as e:
                    # Attempt partial repair
                    try:
                        if json_text.strip().startswith("[") and not json_text.strip().endswith("]"):
                            last_brace = json_text.rfind("}")
                            if last_brace > 0:
                                fixed_json = json_text[:last_brace + 1] + "]"
                                parsed = json.loads(fixed_json)
                                items = parsed if isinstance(parsed, list) else [parsed]
                                print("Successfully parsed partial JSON array")
                            else:
                                raise json.JSONDecodeError("No complete objects found", json_text, 0)
                        elif cfg.mode == "single" and json_text.strip().startswith("{"):
                            parsed = json.loads(json_text)
                            items = [parsed] if isinstance(parsed, dict) else parsed
                            print("Successfully parsed single JSON object")
                        else:
                            raise e
                    except json.JSONDecodeError as e2:
                        print(f"Failed to parse JSON response (variant {variant_idx+1}): {e2}")
                        continue

                # Build overlays and polygons
                W, H = orig.size
                overlay_entries_full: List[Dict[str, Any]] = []
                overlay_entries_box: List[Dict[str, Any]] = []
                polygons_with_labels: List[Tuple[np.ndarray, str]] = []

                for it in items:
                    if not isinstance(it, dict):
                        continue
                    label = it.get("label")
                    mask_uri = it.get("mask")
                    box_2d = it.get("box_2d")
                    if not mask_uri or not box_2d:
                        continue
                    try:
                        mask_im: Image.Image = decode_data_uri_png(mask_uri)
                    except Exception:
                        continue

                    mask_full = mask_im.convert("L").resize((W, H), Image.NEAREST)
                    mask_np = np.array(mask_full)
                    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

                    mapped_label = _choose_class_name(label, cfg.class_names)
                    if cfg.class_names and not any(c.lower() in (label or "").lower() for c in cfg.class_names):
                        continue

                    polys = mask_to_polygons(binary, min_area=cfg.min_area, approx_eps_frac=cfg.approx_eps_frac, export_all=cfg.export_all)
                    if not polys:
                        continue

                    if isinstance(box_2d, list) and len(box_2d) == 4:
                        ymin, xmin, ymax, xmax = box_2d
                    else:
                        ymin = xmin = 0; ymax = 1000; xmax = 1000

                    box_normalized = [xmin / 1000, ymin / 1000, xmax / 1000, ymax / 1000]
                    overlay_entries_full.append({"label": mapped_label, "box_2d": box_normalized, "mask_image": mask_full})
                    overlay_entries_box.append({"label": mapped_label, "box_2d": box_2d, "mask_image": mask_im})
                    for p in polys:
                        polygons_with_labels.append((p, mapped_label))

                if not polygons_with_labels:
                    print(f"No valid polygons for {img_path} (variant {variant_idx+1}/{num_variants})")
                    continue

                # Build JSON
                if cfg.mask_interpretation == "full":
                    polygons_with_labels = []
                    for e in overlay_entries_full:
                        mf = e["mask_image"].convert("L")
                        arr = np.array(mf)
                        cnts, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            area = cv2.contourArea(c)
                            if area < float(cfg.min_area):
                                continue
                            peri = cv2.arcLength(c, True)
                            eps = max(1.0, cfg.approx_eps_frac * peri)
                            approx = cv2.approxPolyDP(c, eps, True)
                            if approx.shape[0] < 3:
                                continue
                            poly = approx.reshape(-1, 2)
                            polygons_with_labels.append((poly, e["label"]))

                v7 = format_v7_polygon_annotations(img_path, polygons_with_labels)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                jpath = os.path.join(ann_dir, f"{base_name}_v{variant_idx+1}.json")
                with open(jpath, "w") as f:
                    json.dump(v7, f, indent=2)
                json_paths.append(jpath)
                uploaded_image_paths.append(img_path)

                # Previews
                try:
                    from vision.visualize import draw_boxes

                    preview_full = orig.convert("RGB")
                    boxes_for_display_full = []
                    for entry in overlay_entries_full:
                        bn = entry["box_2d"]
                        if isinstance(bn, list) and len(bn) == 4:
                            boxes_for_display_full.append({"label": entry["label"], "box_2d": bn})
                    if boxes_for_display_full:
                        from PIL import ImageDraw
                        tmp = preview_full.convert("RGBA").copy()
                        draw = ImageDraw.Draw(tmp)
                        Wp, Hp = tmp.size
                        for b in boxes_for_display_full:
                            from vision.visualize import _convert_box_to_pixels
                            x0,y0,x1,y1 = _convert_box_to_pixels(b["box_2d"], (Wp, Hp))
                            draw.rectangle([(x0,y0),(x1,y1)], outline=(34,197,94,255), width=3)
                        preview_full = tmp.convert("RGB")

                    preview_full = _draw_polygons_outline(preview_full, polygons_with_labels and [p for p,_ in polygons_with_labels] or [], (239,68,68), width=3)

                    preview_box = orig.convert("RGB")
                    boxes_for_display_box = []
                    for entry in overlay_entries_full:
                        bn = entry["box_2d"]
                        if isinstance(bn, list) and len(bn) == 4:
                            boxes_for_display_box.append({"label": entry["label"], "box_2d": bn})
                    if boxes_for_display_box:
                        from PIL import ImageDraw
                        tmp2 = preview_box.convert("RGBA").copy()
                        draw2 = ImageDraw.Draw(tmp2)
                        Wp, Hp = tmp2.size
                        for b in boxes_for_display_box:
                            from vision.visualize import _convert_box_to_pixels
                            x0,y0,x1,y1 = _convert_box_to_pixels(b["box_2d"], (Wp, Hp))
                            draw2.rectangle([(x0,y0),(x1,y1)], outline=(59,130,246,255), width=3)
                        preview_box = tmp2.convert("RGB")

                    try:
                        mask_outline_color = (239, 68, 68)
                        dr = ImageDraw.Draw(preview_box)
                        for e in overlay_entries_box:
                            bn = e["box_2d"]
                            if isinstance(bn, list) and len(bn) == 4:
                                ymin, xmin, ymax, xmax = bn
                                x0 = int(xmin/1000*W); y0 = int(ymin/1000*H)
                                x1 = int(xmax/1000*W); y1 = int(ymax/1000*H)
                                bw = max(1, x1-x0); bh = max(1, y1-y0)
                                raw = e["mask_image"].convert("L").resize((bw,bh), Image.NEAREST)
                                arr = np.array(raw)
                                contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in contours:
                                    if cv2.contourArea(cnt) < 5:
                                        continue
                                    pts = [(int(x0 + p[0][0]), int(y0 + p[0][1])) for p in cnt]
                                    if len(pts) >= 2:
                                        dr.line(pts + [pts[0]], fill=mask_outline_color, width=2)
                    except Exception:
                        pass

                    out_prefix = os.path.join(ann_dir, f"{base_name}_v{variant_idx+1}_preview")
                    if cfg.preview_count >= 1:
                        preview_box.convert("RGB").save(f"{out_prefix}_box.jpg", quality=90)
                    if cfg.preview_count >= 2:
                        preview_full.convert("RGB").save(f"{out_prefix}_full.jpg", quality=90)
                    if cfg.preview_count >= 3:
                        orig.convert("RGB").save(f"{out_prefix}_orig.jpg", quality=90)
                except Exception as e:
                    if cfg.debug:
                        print(f"Failed to create preview for variant {variant_idx+1}: {e}")
                    pass

                # Copy original image to annotations if local-only (once is fine)
                if cfg.local_only and variant_idx == 0:
                    img_filename = os.path.basename(img_path)
                    img_dest = os.path.join(ann_dir, img_filename)
                    try:
                        shutil.copy2(img_path, img_dest)
                    except Exception:
                        pass

                print(f"Annotated {img_path} (v{variant_idx+1}) with {len(polygons_with_labels)} polygon(s)")
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
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
        return 0

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
    return 0

prompt_override = 'Give the segmentation masks for the ink in the image. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels.'

if __name__ == "__main__":
    # Edit the CONFIG values below to control behavior instead of passing CLI args.
    CONFIG = SegAutoConfig(
        # path="../../data/example_images/paint_image1.jpg",
        path="../../data/example_images/paint_image1.jpg",
        items=None,  # Match the image content
        prompt_prefix=None,  # Match the image content
        prompt_override=prompt_override,  # Example: "Give me a single JSON object with the bird's mask and bounding box."
        class_names=["ink"],
        # class_names=["bird"],  # Match the image content
        model="gemini-2.5-flash",
        temperature=DEFAULT_TEMPERATURE,
        max_input_size=DEFAULT_MAX_INPUT_SIZE,
        # Segmentation mode
        mode="single",  # Try "single" for simpler JSON (one object instead of array)
        min_area=100,
        approx_eps_frac=0.01,
        export_all=False,
        debug=False,  # Enable debug to see model responses
        local_only=True,
        v7_team="clearobject",
        v7_dataset="meyers-rsp",
    )
    raise SystemExit(run(CONFIG))


