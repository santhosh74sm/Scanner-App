"""Final Production Validation Script for FastAPI Backend Endpoints."""
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app


def generate_image_bytes(width: int, height: int) -> bytes:
    np.random.seed(42)
    y, x = np.ogrid[:height, :width]
    bg = (200 + 40 * (x / width) + 10 * np.sin(y / 50.0)).astype(np.uint8)
    image = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    line_color = (30, 30, 30)
    for i in range(100, height - 100, 80):
        cv2.line(image, (100, i), (width - 100, i), line_color, 4)

    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes()


def validate_workflow(resolution_name: str, width: int, height: int):
    client = TestClient(app)
    gc.collect()

    print(f"\n==================================================")
    print(f"VALIDATING WORKFLOW: {resolution_name} ({width}x{height})")
    print(f"==================================================")

    # 1. Upload
    image_bytes = generate_image_bytes(width, height)
    file_payload = ("test_doc.jpg", image_bytes, "image/jpeg")

    tracemalloc.start()
    t0 = time.perf_counter()
    res_upload = client.post("/upload", files={"file": file_payload})
    t_upload = (time.perf_counter() - t0) * 1000.0
    _, peak_upload = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
    upload_data = res_upload.json()
    session_id = upload_data["session_id"]
    print(f"[OK] Upload:   {t_upload:6.1f} ms | RAM Peak: {peak_upload / 1e6:5.1f} MB | URL: {upload_data['image_url']}")

    # 2. Serve Source Image (Verify WebP & Cache-Control header)
    res_file = client.get(upload_data["image_url"])
    assert res_file.status_code == 200
    assert "image/webp" in res_file.headers.get("content-type", "")
    assert "public, max-age=86400, immutable" in res_file.headers.get("cache-control", "")
    print(f"[OK] Serve File: Content-Type={res_file.headers['content-type']}, Cache-Control={res_file.headers['cache-control']}, Size={len(res_file.content)/1e3:.1f} KB")

    # 3. Detect
    tracemalloc.start()
    t0 = time.perf_counter()
    res_detect = client.post(f"/detect?session_id={session_id}")
    t_detect = (time.perf_counter() - t0) * 1000.0
    _, peak_detect = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    assert res_detect.status_code == 200, f"Detect failed: {res_detect.text}"
    detect_data = res_detect.json()
    corners = detect_data["corners"]
    print(f"[OK] Detect:   {t_detect:6.1f} ms | RAM Peak: {peak_detect / 1e6:5.1f} MB | Corners: {len(corners)} points")

    # 4. Crop
    crop_payload = {"session_id": session_id, "corners": corners}
    tracemalloc.start()
    t0 = time.perf_counter()
    res_crop = client.post("/crop", json=crop_payload)
    t_crop = (time.perf_counter() - t0) * 1000.0
    _, peak_crop = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    assert res_crop.status_code == 200, f"Crop failed: {res_crop.text}"
    crop_data = res_crop.json()
    print(f"[OK] Crop:     {t_crop:6.1f} ms | RAM Peak: {peak_crop / 1e6:5.1f} MB | URL: {crop_data['image_url']}")

    # 5. Enhance
    enhance_payload = {"session_id": session_id, "mode": "black_white"}
    tracemalloc.start()
    t0 = time.perf_counter()
    res_enhance = client.post("/enhance", json=enhance_payload)
    t_enhance = (time.perf_counter() - t0) * 1000.0
    _, peak_enhance = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    assert res_enhance.status_code == 200, f"Enhance failed: {res_enhance.text}"
    enhance_data = res_enhance.json()
    print(f"[OK] Enhance:  {t_enhance:6.1f} ms | RAM Peak: {peak_enhance / 1e6:5.1f} MB | URL: {enhance_data['image_url']}")

    # 6. Download PNG
    res_dl_png = client.get(f"/download?session_id={session_id}&format=png")
    assert res_dl_png.status_code == 200
    assert "image/png" in res_dl_png.headers["content-type"]
    print(f"[OK] Download PNG: Size={len(res_dl_png.content)/1e3:.1f} KB | Content-Type={res_dl_png.headers['content-type']}")

    # 7. Download JPG
    res_dl_jpg = client.get(f"/download?session_id={session_id}&format=jpg")
    assert res_dl_jpg.status_code == 200
    assert "image/jpeg" in res_dl_jpg.headers["content-type"]
    print(f"[OK] Download JPG: Size={len(res_dl_jpg.content)/1e3:.1f} KB | Content-Type={res_dl_jpg.headers['content-type']}")

    max_ram = max(peak_upload, peak_detect, peak_crop, peak_enhance) / 1e6
    print(f"--> MAX WORKFLOW PEAK RAM: {max_ram:.1f} MB (Render Limit: 512.0 MB)")


def main():
    resolutions = [
        ("2MP", 1920, 1080),
        ("8MP", 3264, 2448),
        ("12MP", 4000, 3000),
        ("20MP", 5000, 4000),
    ]

    print("STARTING END-TO-END PRODUCTION VALIDATION TEST SUITE...")
    for name, w, h in resolutions:
        validate_workflow(name, w, h)
    print("\nALL WORKFLOW VALIDATIONS PASSED SUCCESSFULLY WITH ZERO ERRORS!")


if __name__ == "__main__":
    main()
