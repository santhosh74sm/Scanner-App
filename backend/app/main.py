"""FastAPI entrypoint for the document scanner application."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .models.schemas import CornersRequest, EnhanceRequest
from .scanner.service import ScannerService

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SESSION_MAX_AGE_SECONDS = 86400  # 24 hours
MAX_CACHED_SESSIONS = 4

MEDIA_TYPES = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

service = ScannerService()


class SessionImageCache:
    """Bounded cache for active scan sessions.

    Disk remains the recovery source after a restart or eviction, while the
    normal interactive path reuses decoded matrices and encoded response bytes.
    """

    def __init__(self, max_sessions: int = MAX_CACHED_SESSIONS) -> None:
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, dict[str, object]] = OrderedDict()

    def get(self, session_id: str, name: str) -> np.ndarray | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        self.sessions.move_to_end(session_id)
        image = session.get(name)
        return image if isinstance(image, np.ndarray) else None

    def get_bytes(self, session_id: str, name: str) -> bytes | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        self.sessions.move_to_end(session_id)
        payload = session.get(f"{name}_bytes")
        return payload if isinstance(payload, bytes) else None

    def put(self, session_id: str, name: str, image: np.ndarray, payload: bytes) -> None:
        session = self.sessions.setdefault(session_id, {})
        session[name] = image
        session[f"{name}_bytes"] = payload
        self.sessions.move_to_end(session_id)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)

    def put_bytes(self, session_id: str, name: str, payload: bytes) -> None:
        session = self.sessions.setdefault(session_id, {})
        session[f"{name}_bytes"] = payload
        self.sessions.move_to_end(session_id)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)

    def discard(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


session_cache = SessionImageCache()

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
                    session_cache.discard(item.name)
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


def read_session_image(session_id: str, directory: Path, name: str) -> np.ndarray:
    cached = session_cache.get(session_id, name)
    if cached is not None:
        return cached
    jpg_path = directory / f"{name}.jpg"
    webp_path = directory / f"{name}.webp"
    png_path = directory / f"{name}.png"
    if jpg_path.exists():
        return read_image(jpg_path)
    elif webp_path.exists():
        return read_image(webp_path)
    elif png_path.exists():
        return read_image(png_path)
    raise HTTPException(404, f"Session image {name} not found.")


def encode_image(path: Path, image: np.ndarray, quality: int = 85) -> bytes:
    ext = path.suffix.lower()
    params = []
    if ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    success, encoded = cv2.imencode(path.suffix.lower(), image, params)
    if not success:
        raise HTTPException(500, f"Unable to encode image file {path.name}.")
    return encoded.tobytes()


def make_preview_image(image: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def persist_image(session_id: str, path: Path, name: str, image: np.ndarray, quality: int = 85) -> bytes:
    # 1. Save full-resolution high-quality JPEG to disk for full-res storage & cache recovery
    jpg_path = path.with_suffix(".jpg")
    full_payload = encode_image(jpg_path, image, quality=95)
    jpg_path.write_bytes(full_payload)

    # 2. Save lightweight display preview WebP for fast browser UI rendering
    preview_img = make_preview_image(image, max_dim=1600)
    webp_path = path.with_suffix(".webp")
    preview_payload = encode_image(webp_path, preview_img, quality=quality)
    webp_path.write_bytes(preview_payload)

    # 3. Cache full-resolution matrix + preview bytes in session cache
    session_cache.put(session_id, name, image, preview_payload)
    return preview_payload


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

    persist_image(session_id, directory / "source.webp", "source", image, quality=85)
    height, width = image.shape[:2]

    # Pre-compute document detection corners during upload to save a network roundtrip
    corners = service.detect(image).round(2).tolist()

    return {
        "session_id": session_id,
        "image_url": f"/files/{session_id}/source.webp",
        "width": width,
        "height": height,
        "corners": corners,
    }


@app.post("/detect")
def detect(session_id: str) -> dict:
    directory = session_dir(session_id)
    source_img = read_session_image(session_id, directory, "source")
    corners = service.detect(source_img).round(2).tolist()
    return {"session_id": session_id, "corners": corners}


@app.post("/crop")
def crop(payload: CornersRequest) -> dict:
    directory = session_dir(payload.session_id)
    source_img = read_session_image(payload.session_id, directory, "source")
    try:
        cropped = service.crop(source_img, payload.corners)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    persist_image(payload.session_id, directory / "cropped.webp", "cropped", cropped, quality=85)
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
        image_to_enhance = read_session_image(payload.session_id, directory, "cropped")
    except HTTPException:
        image_to_enhance = read_session_image(payload.session_id, directory, "source")

    try:
        final = service.enhance(image_to_enhance, payload.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    persist_image(payload.session_id, directory / "final.webp", "final", final, quality=85)
    return {
        "session_id": payload.session_id,
        "mode": payload.mode,
        "image_url": f"/files/{payload.session_id}/final.webp",
    }


@app.get("/download")
def download(session_id: str, format: str = "png") -> Response:
    if format not in {"png", "jpg"}:
        raise HTTPException(422, "Format parameter must be 'png' or 'jpg'.")

    directory = session_dir(session_id)
    try:
        image = read_session_image(session_id, directory, "final")
    except HTTPException:
        try:
            image = read_session_image(session_id, directory, "cropped")
        except HTTPException:
            image = read_session_image(session_id, directory, "source")

    cached = session_cache.get_bytes(session_id, f"scan_{format}")
    if cached is None:
        cached = encode_image(directory / f"scan.{format}", image, quality=95)
        session_cache.put_bytes(session_id, f"scan_{format}", cached)

    media_type = "image/png" if format == "png" else "image/jpeg"
    return Response(
        content=cached,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="document-scan.{format}"'},
    )


@app.get("/files/{session_id}/{name}")
def serve_file(session_id: str, name: str) -> Response:
    if name not in {
        "source.webp", "cropped.webp", "final.webp",
        "source.png", "cropped.png", "final.png",
        "source.jpg", "cropped.jpg", "final.jpg",
    }:
        raise HTTPException(404, "Requested resource not found.")

    directory = session_dir(session_id)
    stem = Path(name).stem
    cached = session_cache.get_bytes(session_id, stem)
    if cached is not None:
        media_type = MEDIA_TYPES.get(Path(name).suffix.lower(), "image/png")
        return Response(
            content=cached,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )
    file_path = directory / name
    if not file_path.exists():
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
