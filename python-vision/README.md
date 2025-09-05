# Python Vision

A comprehensive computer vision toolkit with AI-powered annotation capabilities.

## Features

- 🤖 **AI-Powered Detection**: Pure-Python package with Gemini integration
- 📦 **Multiple Modes**: 2D boxes, Segmentation, Points, 3D boxes
- 🚀 **FastAPI Server**: Clean backend API for web interfaces
- 🖼️ **Auto Annotation**: Batch image annotation with V7 Darwin integration
- 🛠️ **CLI Tools**: Command-line interface for direct usage
- 🌐 **Web UI**: Separate React frontend (see web-ui folder)

## Components

### Core Vision Package (`vision/`)
- Python package with Gemini AI integration
- Supports multiple detection modes
- Configurable models and parameters

### Auto Annotate (`auto-annotate/`)
- Automated bounding box detection using Gemini AI
- Batch processing for images and directories
- V7 Darwin annotation platform integration
- Local mode for offline annotation

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Setup

Set your API keys:
```bash
# Required for AI features
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional for V7 annotation upload
export V7_API_KEY="your-v7-api-key"
```

## Usage

### Vision CLI
```bash
# Basic object detection
python -m vision.cli --mode boxes --image ../data/example_images/car2.jpeg --max-input-size 640 --output-pixels

# Segmentation
python -m vision.cli --mode segmentation --image image.jpg

# Point detection
python -m vision.cli --mode points --image image.jpg
```

### Auto Annotation
```bash
# Single image
python auto-annotate/auto_annotate_bbox.py --path image.jpg --items "cars"

# Batch processing with V7 upload
python auto-annotate/auto_annotate_bbox.py \
  --path ./images/ \
  --items "vehicles" \
  --v7_team "your-team" \
  --v7_dataset "vehicle-detection"

# Local-only mode
python auto-annotate/auto_annotate_bbox.py \
  --path ./images/ \
  --items "products" \
  --local-only
```

### API Server
```bash
uvicorn vision.api:app --reload --port 8010
```

### Web UI
See `web-ui/README.md` for running the React frontend that connects to the API.

## Project Structure

```
python-vision/
├── vision/                 # Core vision package
│   ├── api.py             # FastAPI server
│   ├── cli.py             # Command-line interface
│   ├── client.py          # Gemini client setup
│   ├── config.py          # Configuration and model settings
│   ├── io_utils.py        # Image loading and processing
│   ├── modes.py           # Detection modes (boxes, segmentation, etc.)
│   ├── prompts.py         # AI prompt templates
│   └── visualize.py       # Visualization utilities
├── auto-annotate/         # Auto annotation tools
│   ├── auto_annotate_bbox.py  # Main annotation script
│   └── README.md          # Detailed usage guide
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

- Python 3.8+
- GOOGLE_API_KEY environment variable
- V7_API_KEY (optional, for annotation upload)
- PIL (Pillow)
- FastAPI
- Uvicorn
- python-dotenv
- darwin-py (for V7 integration)
