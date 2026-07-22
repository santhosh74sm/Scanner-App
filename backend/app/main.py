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

MEDIA_TYPES = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

service = ScannerService()

app = FastAPI(
    title="Document Scanner API",
    description="High-precision document boundary detection, perspective transform, and enhancement API.",
    version="1.0.0",
)

# CORS Fix: allow_credentials=False with wildcard origins to prevent Mobile Safari / Chrome CORS preflight failures
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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


def read_session_image(directory: Path, name: str) -> np.ndarray:
    webp_path = directory / f"{name}.webp"
    png_path = directory / f"{name}.png"
    jpg_path = directory / f"{name}.jpg"
    if webp_path.exists():
        return read_image(webp_path)
    elif png_path.exists():
        return read_image(png_path)
    elif jpg_path.exists():
        return read_image(jpg_path)
    raise HTTPException(404, f"Session image {name} not found.")


def write_image(path: Path, image: np.ndarray, quality: int = 85) -> None:
    ext = path.suffix.lower()
    params = []
    if ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    success = cv2.imwrite(str(path), image, params)
    if not success:
        # Fallback to PNG if format encoding fails
        fallback_path = path.with_suffix(".png")
        if not cv2.imwrite(str(fallback_path), image):
            raise HTTPException(500, f"Unable to write image file {path.name}.")


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

    # Save lightweight WebP for web previews (~800KB vs ~12MB)
    write_image(directory / "source.webp", image, quality=85)
    height, width = image.shape[:2]

    return {
        "session_id": session_id,
        "image_url": f"/files/{session_id}/source.webp",
        "width": width,
        "height": height,
    }


@app.post("/detect")
def detect(session_id: str) -> dict:
    directory = session_dir(session_id)
    source_img = read_session_image(directory, "source")
    corners = service.detect(source_img).round(2).tolist()
    return {"session_id": session_id, "corners": corners}


@app.post("/crop")
def crop(payload: CornersRequest) -> dict:
    directory = session_dir(payload.session_id)
    source_img = read_session_image(directory, "source")
    try:
        cropped = service.crop(source_img, payload.corners)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    write_image(directory / "cropped.webp", cropped, quality=85)
    height, width = cropped.shape[:2]

    return {
        "session_id": payload.session_id,
        "image_url": f"/files/{payload.session_id}/cropped.webp",
        "width": width,
        "height": height,
    }


@app.post("/enhance")
def enhance(payload: EnhanceRequest) -> dict:
    directory = session_dir(payload.session_id)
    try:
        image_to_enhance = read_session_image(directory, "cropped")
    except HTTPException:
        image_to_enhance = read_session_image(directory, "source")

    try:
        final = service.enhance(image_to_enhance, payload.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    write_image(directory / "final.webp", final, quality=85)
    return {
        "session_id": payload.session_id,
        "mode": payload.mode,
        "image_url": f"/files/{payload.session_id}/final.webp",
    }


@app.get("/download")
def download(session_id: str, format: str = "png") -> FileResponse:
    if format not in {"png", "jpg"}:
        raise HTTPException(422, "Format parameter must be 'png' or 'jpg'.")

    directory = session_dir(session_id)
    try:
        image = read_session_image(directory, "final")
    except HTTPException:
        try:
            image = read_session_image(directory, "cropped")
        except HTTPException:
            image = read_session_image(directory, "source")

    output_file = directory / f"scan.{format}"
    write_image(output_file, image, quality=95)

    media_type = "image/png" if format == "png" else "image/jpeg"
    return FileResponse(output_file, media_type=media_type, filename=f"document-scan.{format}")


@app.get("/files/{session_id}/{name}")
def serve_file(session_id: str, name: str) -> FileResponse:
    if name not in {"source.webp", "cropped.webp", "final.webp", "source.png", "cropped.png", "final.png"}:
        raise HTTPException(404, "Requested resource not found.")

    directory = session_dir(session_id)
    file_path = directory / name
    if not file_path.exists():
        stem = Path(name).stem
        alt_webp = directory / f"{stem}.webp"
        alt_png = directory / f"{stem}.png"
        if alt_webp.exists():
            file_path = alt_webp
        elif alt_png.exists():
            file_path = alt_png
        else:
            raise HTTPException(404, "Requested image file is not ready.")

    ext = file_path.suffix.lower()
    media_type = MEDIA_TYPES.get(ext, "image/png")
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
