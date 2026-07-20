"""Service layer wrapping the OpenCV document scanner engine and enhancer."""
from __future__ import annotations

import cv2
import numpy as np

from .doc_scanner import DocScanner
from .enhancer import DocumentEnhancer
from .utils.transform import four_point_transform, order_points


class ScannerService:
    """Detect, rectify, and enhance document images."""

    def __init__(self) -> None:
        self.detector = DocScanner(interactive=False)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return four document corner points in original-image pixel coordinates."""
        scale = image.shape[0] / 500.0
        preview = cv2.resize(image, (max(1, round(image.shape[1] / scale)), 500))
        return self.detector.get_contour(preview) * scale

    @staticmethod
    def crop(image: np.ndarray, corners: list[list[float]]) -> np.ndarray:
        points = np.asarray(corners, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("Exactly four [x, y] corners are required.")
        if not cv2.isContourConvex(order_points(points).reshape(-1, 1, 2)):
            raise ValueError("Crop corners must form a convex quadrilateral.")
        return four_point_transform(image, points)

    @staticmethod
    def enhance(image: np.ndarray, mode: str) -> np.ndarray:
        """Delegate enhancement to the production-grade DocumentEnhancer module."""
        return DocumentEnhancer.process(image, mode)
