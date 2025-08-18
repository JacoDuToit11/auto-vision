#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI backend
PORT=${PORT:-8010}
API_CMD="uvicorn vision.api:app --port ${PORT} --reload"

# Start Vite frontend
pushd web-ui >/dev/null
if [ ! -f node_modules/.package-lock.json ] && [ ! -d node_modules ]; then
	npm install
fi
popd >/dev/null

# Open browser after a short delay (Vite dev server runs on 5173)
sleep 2
(open "http://localhost:5173" || true) >/dev/null 2>&1 &

# Run both in parallel
( cd python-vision && ${API_CMD} ) &
BACK_PID=$!
( cd web-ui && npm run dev ) &
FRONT_PID=$!

# Wait for either to exit
wait $BACK_PID $FRONT_PID
