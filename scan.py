"""Scan photographed documents using the OpenCV Document Scanner engine."""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.scanner.doc_scanner import DocScanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and scan photographed documents.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--images", type=Path, help="Directory of images to scan")
    sources.add_argument("--image", type=Path, help="Single image to scan")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory (default: output)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scanner = DocScanner()
    formats = {".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"}

    if args.image:
        scanner.scan(args.image, args.output_dir)
        return

    if not args.images.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {args.images}")

    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in formats)
    if not images:
        raise ValueError(f"No supported images found in: {args.images}")

    for image in images:
        scanner.scan(image, args.output_dir)


if __name__ == "__main__":
    main()
