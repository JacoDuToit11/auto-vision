# Auto Annotate

A powerful script that uses Google's Gemini AI to automatically detect objects in images and generate bounding box annotations, with optional upload to V7 Darwin annotation platform.

## Features

- 🖼️ **Batch Processing**: Process single images or entire directories
- 🤖 **AI-Powered Detection**: Uses Gemini 2.5 Pro for accurate object detection
- 📦 **V7 Integration**: Seamless upload to V7 Darwin annotation platform
- 🔧 **Flexible Configuration**: Customizable detection prompts and filtering
- 🏠 **Local Mode**: Generate annotations locally without uploading
- 🐛 **Debug Previews**: Visual previews of detected bounding boxes
- 🎯 **Class Filtering**: Filter detections by specific class names

## Installation

1. Install the vision package:
```bash
pip install -r ../requirements.txt
```

2. Set up environment variables:
```bash
# Required for Gemini AI
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional for V7 upload
export V7_API_KEY="your-v7-api-key"
```

## Usage

### Basic Usage

Detect objects in a single image:
```bash
python auto_annotate_bbox.py --path image.jpg --items "cars"
```

Process all images in a directory:
```bash
python auto_annotate_bbox.py --path ./images/ --items "cars"
```

### Advanced Usage

With custom classes and V7 upload:
```bash
python auto_annotate_bbox.py \
  --path ./images/ \
  --items "vehicles including cars, trucks, and motorcycles" \
  --class_names "car" "truck" "motorcycle" \
  --v7_team "your-team" \
  --v7_dataset "vehicle-detection" \
  --model "gemini-2.5-pro" \
  --temperature 0.1
```

Local-only mode (no upload):
```bash
python auto_annotate_bbox.py \
  --path ./images/ \
  --items "paint cans" \
  --local-only
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--path` | `../../data/example_images/paint_image1.jpg` | Image file or directory path |
| `--items` | `paint` | What to detect (e.g., "cars", "people", "buildings") |
| `--class_names` | `["paint"]` | Specific class names to filter for |
| `--v7_team` | `clearobject` | V7 Team Slug |
| `--v7_dataset` | `human-in-office-detection` | V7 Dataset Slug |
| `--model` | `gemini-2.5-pro` | Gemini model to use |
| `--temperature` | `0.1` | Model temperature (0.0-1.0) |
| `--max-input-size` | `1024` | Maximum input image size |
| `--min-box` | `24` | Minimum box size in normalized units |
| `--debug` | `False` | Show debug information |
| `--local-only` | `False` | Skip V7 upload, keep local annotations |

## Output

### Local Mode (`--local-only`)
- Creates `v7_upload_gemini/annotations/` directory
- Saves V7-compatible JSON annotation files
- Copies original images
- Generates debug preview images with bounding boxes

### V7 Upload Mode
- Uploads images to V7 Darwin
- Imports annotations automatically
- Handles duplicate detection and retry logic

## Annotation Format

The script generates V7 Darwin-compatible JSON annotations:

```json
{
  "version": "2.0",
  "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
  "item": {
    "name": "image.jpg",
    "path": "/"
  },
  "annotations": [
    {
      "bounding_box": {
        "x": 100.0,
        "y": 150.0,
        "w": 200.0,
        "h": 180.0
      },
      "id": "uuid-here",
      "name": "car",
      "slot_names": ["0"]
    }
  ]
}
```

## Tips

1. **Prompt Engineering**: Use descriptive prompts like "vehicles including cars, trucks, and motorcycles" for better results
2. **Class Filtering**: Use `--class_names` to filter detections to specific classes
3. **Batch Processing**: Process directories for efficient bulk annotation
4. **Debug Mode**: Use `--debug` to see raw AI responses and troubleshoot issues
5. **Local Testing**: Use `--local-only` first to verify detections before uploading

## Requirements

- Python 3.8+
- GOOGLE_API_KEY environment variable
- V7_API_KEY (optional, only for upload)
- PIL (Pillow)
- python-dotenv
- darwin-py (for V7 integration)

## Troubleshooting

### Common Issues

1. **No boxes detected**: Try adjusting the prompt or reducing `--min-box` size
2. **V7 upload fails**: Check V7_API_KEY and team/dataset permissions
3. **Memory errors**: Reduce `--max-input-size` for large images
4. **Class filtering too strict**: Verify class names match detection labels

### Debug Information

Use `--debug` flag to see:
- Raw AI model responses
- Box coordinates and labels
- Processing pipeline details

## Examples

### Detect People in Office Images
```bash
python auto_annotate_bbox.py \
  --path ./office_photos/ \
  --items "people working at desks or in meetings" \
  --class_names "person" \
  --v7_dataset "office-occupancy"
```

### Local Product Detection
```bash
python auto_annotate_bbox.py \
  --path ./products/ \
  --items "consumer products on shelves" \
  --local-only \
  --debug
```

### Custom Model Settings
```bash
python auto_annotate_bbox.py \
  --path image.jpg \
  --model "gemini-2.5-pro" \
  --temperature 0.0 \
  --max-input-size 2048
```
