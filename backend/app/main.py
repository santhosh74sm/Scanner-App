"""FastAPI entrypoint for the document scanner application."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models.schemas import CornersRequest, EnhanceRequest
from .scanner.service import ScannerService

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SESSION_MAX_AGE_SECONDS = 86400  # 24 hours

service = ScannerService()

app = FastAPI(
    title="Document Scanner API",
    description="High-precision document boundary detection, perspective transform, and enhancement API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cleanup_stale_sessions() -> None:
    """Remove session directories older than SESSION_MAX_AGE_SECONDS."""
    now = time.time()
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            try:
                if now - item.stat().st_mtime > SESSION_MAX_AGE_SECONDS:
                    for subfile in item.iterdir():
                        subfile.unlink(missing_ok=True)
                    item.rmdir()
            except OSError:
                pass


def session_dir(session_id: str) -> Path:
    path = DATA_DIR / session_id
    if not path.is_dir():
        raise HTTPException(404, "Scan session not found.")
    return path


def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise HTTPException(404, f"Image path {path.name} does not exist.")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "The file could not be decoded as an image.")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise HTTPException(500, "Unable to write image file.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    cleanup_stale_sessions()
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported file extension. Upload a JPG, PNG, WEBP, BMP, or TIFF image.")
    
    content = await file.read()
    await file.close()
    
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image size must be between 1 byte and {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "The uploaded file could not be decoded as a valid image.")
    
    session_id = uuid.uuid4().hex
    directory = DATA_DIR / session_id
    directory.mkdir(parents=True, exist_ok=True)
    
    write_image(directory / "source.png", image)
    height, width = image.shape[:2]
    
    return {
        "session_id": session_id,
        "image_url": f"/files/{session_id}/source.png",
        "width": width,
        "height": height,
    }


@app.post("/detect")
def detect(session_id: str) -> dict:
    directory = session_dir(session_id)
    source_img = read_image(directory / "source.png")
    corners = service.detect(source_img).round(2).tolist()
    return {"session_id": session_id, "corners": corners}


@app.post("/crop")
def crop(payload: CornersRequest) -> dict:
    directory = session_dir(payload.session_id)
    source_img = read_image(directory / "source.png")
    try:
        cropped = service.crop(source_img, payload.corners)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    
    write_image(directory / "cropped.png", cropped)
    height, width = cropped.shape[:2]
    
    return {
        "session_id": payload.session_id,
        "image_url": f"/files/{payload.session_id}/cropped.png",
        "width": width,
        "height": height,
    }


@app.post("/enhance")
def enhance(payload: EnhanceRequest) -> dict:
    directory = session_dir(payload.session_id)
    cropped_file = directory / "cropped.png"
    if not cropped_file.exists():
        cropped_file = directory / "source.png"
        if not cropped_file.exists():
            raise HTTPException(409, "Crop or upload the document before applying enhancement.")
    
    image_to_enhance = read_image(cropped_file)
    try:
        final = service.enhance(image_to_enhance, payload.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    
    write_image(directory / "final.png", final)
    return {
        "session_id": payload.session_id,
        "mode": payload.mode,
        "image_url": f"/files/{payload.session_id}/final.png",
    }


@app.get("/download")
def download(session_id: str, format: str = "png") -> FileResponse:
    if format not in {"png", "jpg"}:
        raise HTTPException(422, "Format parameter must be 'png' or 'jpg'.")
    
    directory = session_dir(session_id)
    target_path = directory / "final.png"
    if not target_path.exists():
        target_path = directory / "cropped.png"
    if not target_path.exists():
        target_path = directory / "source.png"
        
    image = read_image(target_path)
    output_file = directory / f"scan.{format}"
    write_image(output_file, image)
    
    media_type = "image/png" if format == "png" else "image/jpeg"
    return FileResponse(output_file, media_type=media_type, filename=f"document-scan.{format}")


@app.get("/files/{session_id}/{name}")
def serve_file(session_id: str, name: str) -> FileResponse:
    if name not in {"source.png", "cropped.png", "final.png"}:
        raise HTTPException(404, "Requested resource not found.")
    
    file_path = session_dir(session_id) / name
    if not file_path.exists():
        raise HTTPException(404, "Requested image file is not ready.")
    
    return FileResponse(file_path, media_type="image/png")
