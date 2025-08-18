Python Vision

- Pure-Python package and CLI that mirrors the TSX app’s modes (2D boxes, Segmentation, Points, 3D boxes).
- Clean FastAPI server to drive a separate web UI; CLI remains usable.
- Requires GOOGLE_API_KEY in your shell.

Install
  python -m venv .venv && source .venv/bin/activate
  pip install -r python-vision/requirements.txt

CLI
  python -m vision.cli --mode boxes --image ../data/example_images/car2.jpeg --max-input-size 640 --output-pixels

API (serve backend)
  uvicorn vision.api:app --reload --port 8010

Web UI (separate folder)
  See web-ui/README.md for running the frontend (Vite React) that calls the API.
