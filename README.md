# Paperly Document Scanner

A modern, single-image document-scanning application built around the repository's existing OpenCV detector and four-point perspective transform. The original `scan.py` command-line scanner remains available; the web experience layers FastAPI and React on top of the same `DocScanner.get_contour` and `four_point_transform` implementation.

## Features

- One-image workflow: upload → auto-detect → manually adjust four corners → perspective preview → enhance → download.
- Original OpenCV LSD/contour detection and perspective-correction pipeline retained.
- B&W Clean document enhancement.
- Responsive React UI with drag/drop, draggable crop corners, loading states, progress tracking, comparisons, and JPG/PNG downloads.
- Validated FastAPI endpoints and isolated per-scan sessions.

## Run locally

Use Python 3.10+ and Node.js 20+.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8001
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api` to FastAPI on port 8001. API docs are available at `http://127.0.0.1:8001/docs`.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Upload one supported image as multipart field `file`. Returns a session ID and image metadata. |
| `POST /detect?session_id=...` | Run the preserved detector and return four pixel-space corners. |
| `POST /crop` | Send `{ session_id, corners: [[x,y], ...] }` to apply the four-point transform. |
| `POST /enhance` | Send `{ session_id, mode }` after crop to update the final preview. |
| `GET /download?session_id=...&format=png` | Download final scan as `png` or `jpg`. |

Supported upload types: JPEG, PNG, WEBP, BMP, and TIFF. Sessions are filesystem-isolated under `backend/data/` and are intentionally single-image only.

## Existing CLI scanner

The legacy command stays intact:

```powershell
python scan.py --image sample_images/desk.JPG
```

## Verification

- `python -m compileall backend` checks the API package syntax.
- `npm run build` checks the React/TypeScript production build.
- Manual browser validation: upload an image, drag each crop handle, continue through comparison and enhancements, then download both output formats.
