"""Benchmark script to measure peak memory (RAM) and runtime before & after optimization."""
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.scanner.enhancer import DocumentEnhancer
from app.scanner.service import ScannerService


def generate_test_image(width: int, height: int) -> np.ndarray:
    """Generate a realistic test image with paper-like texture and text lines."""
    np.random.seed(42)
    # Base background (paper-like gradient)
    y, x = np.ogrid[:height, :width]
    bg = (200 + 40 * (x / width) + 10 * np.sin(y / 50.0)).astype(np.uint8)
    image = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # Draw text lines & shapes
    line_color = (30, 30, 30)
    for i in range(100, height - 100, 80):
        cv2.line(image, (100, i), (width - 100, i), line_color, 4)
    return image


def benchmark_pipeline(resolution_name: str, width: int, height: int) -> dict:
    """Benchmark full scan & enhance pipeline memory and runtime."""
    gc.collect()
    img = generate_test_image(width, height)

    service = ScannerService()
    corners = [[100, 100], [width - 100, 100], [width - 100, height - 100], [100, height - 100]]

    # Measure Detect
    tracemalloc.start()
    t0 = time.perf_counter()
    detected_corners = service.detect(img)
    t_detect = (time.perf_counter() - t0) * 1000.0
    _, peak_detect = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    # Measure Crop
    tracemalloc.start()
    t0 = time.perf_counter()
    cropped = service.crop(img, corners)
    t_crop = (time.perf_counter() - t0) * 1000.0
    _, peak_crop = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    # Measure Enhance
    tracemalloc.start()
    t0 = time.perf_counter()
    enhanced = service.enhance(cropped, "black_white")
    t_enhance = (time.perf_counter() - t0) * 1000.0
    _, peak_enhance = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    peak_detect_mb = peak_detect / (1024 * 1024)
    peak_crop_mb = peak_crop / (1024 * 1024)
    peak_enhance_mb = peak_enhance / (1024 * 1024)

    return {
        "resolution": resolution_name,
        "dim": f"{width}x{height}",
        "pixels_mp": (width * height) / 1e6,
        "detect": {"time_ms": t_detect, "peak_mb": peak_detect_mb},
        "crop": {"time_ms": t_crop, "peak_mb": peak_crop_mb},
        "enhance": {"time_ms": t_enhance, "peak_mb": peak_enhance_mb},
        "max_peak_mb": max(peak_detect_mb, peak_crop_mb, peak_enhance_mb),
    }


def main():
    resolutions = [
        ("2MP", 1920, 1080),
        ("8MP", 3264, 2448),
        ("12MP", 4000, 3000),
        ("20MP", 5000, 4000),
    ]

    print("=" * 70)
    print("MEMORY AND PERFORMANCE BENCHMARK (AFTER OPTIMIZATION)")
    print("=" * 70)

    for name, w, h in resolutions:
        res = benchmark_pipeline(name, w, h)
        print(f"\n--- {res['resolution']} ({res['dim']}, {res['pixels_mp']:.1f} MP) ---")
        print(f"  Detect:  {res['detect']['time_ms']:6.1f} ms | Peak RAM: {res['detect']['peak_mb']:6.1f} MB")
        print(f"  Crop:    {res['crop']['time_ms']:6.1f} ms | Peak RAM: {res['crop']['peak_mb']:6.1f} MB")
        print(f"  Enhance: {res['enhance']['time_ms']:6.1f} ms | Peak RAM: {res['enhance']['peak_mb']:6.1f} MB")
        print(f"  MAX PEAK RAM: {res['max_peak_mb']:.1f} MB")


if __name__ == "__main__":
    main()
