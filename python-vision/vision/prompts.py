# Prompts derived from copy-of-spatial-understanding/consts.tsx and Prompt.tsx

DEFAULT_PROMPT_PARTS = {
	"boxes": [
		"Show me the positions of",
		"items",
		' as a JSON list. Each item should have "box_2d": [x0, y0, x1, y1] in pixel coordinates and "label": <name>. Do not return masks. Limit to 25 items.',
	],
	"seg": [
		"Give the segmentation masks for",
		"all objects",
		'. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d" as [x0, y0, x1, y1] in pixel coordinates, the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels.',
	],
	"boxes3d": [
		"Output in json. Detect the 3D bounding boxes of",
		"items",
		', output no more than 10 items. Return a list where each entry contains the object name in "label" and its 3D bounding box in "box_3d".',
	],
	"points": [
		"Point to the",
		"items",
		' with no more than 10 items. The answer should follow the json format: [{"point": [x, y], "label": <label1>}, ...]. The points are in pixel coordinates.',
	],
}


def make_prompt(mode: str, items: str | None, seg_language: str | None, max_items: int) -> str:
	if mode not in DEFAULT_PROMPT_PARTS:
		raise ValueError(f"Unknown mode: {mode}")
	p0, p1, p2 = DEFAULT_PROMPT_PARTS[mode]
	target = items if items else p1
	if mode == "seg" and seg_language and seg_language.strip() and seg_language.strip().lower() != "english":
		p2 = p2.replace(
			' text label in the key "label". Use descriptive labels.',
			""
		) + f' text label in language {seg_language} in the key "label". Use descriptive labels in {seg_language}. Ensure labels are in {seg_language}. DO NOT USE ENGLISH FOR LABELS.'
	if mode == "boxes":
		p2 = p2.replace("25", str(max_items))
	return f"{p0} {target} {p2}"
