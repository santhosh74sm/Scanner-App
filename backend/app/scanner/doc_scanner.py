"""Document boundary detection and image scan engine using OpenCV line segment detection."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
import numpy as np

from .utils import imutils, transform


class DocScanner:
    """Detect, rectify, and enhance document boundaries in photos."""

    def __init__(self, interactive: bool = False, min_quad_area_ratio: float = 0.25,
                 max_quad_angle_range: float = 40.0) -> None:
        self.interactive = interactive
        self.min_quad_area_ratio = min_quad_area_ratio
        self.max_quad_angle_range = max_quad_angle_range

    @staticmethod
    def filter_corners(corners: list[tuple[int, int]], min_dist: float = 20.0) -> list[tuple[int, int]]:
        """Keep spatially distinct candidate corner points."""
        filtered: list[tuple[int, int]] = []
        for corner in corners:
            if all(math.dist(existing, corner) >= min_dist for existing in filtered):
                filtered.append(corner)
        return filtered

    @staticmethod
    def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        if denominator == 0:
            return 180.0
        cosine = np.clip(np.dot(u, v) / denominator, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def angle_range(self, quad: np.ndarray) -> float:
        """Return the spread of interior angles of a quadrilateral."""
        tl, tr, br, bl = transform.order_points(np.asarray(quad).reshape(4, 2))
        angles = (
            self._angle_between(bl - tl, tr - tl),
            self._angle_between(tl - tr, br - tr),
            self._angle_between(tr - br, bl - br),
            self._angle_between(br - bl, tl - bl),
        )
        return float(np.ptp(angles))

    def get_corners(self, edges: np.ndarray) -> list[tuple[int, int]]:
        """Extract document corner candidates using OpenCV LineSegmentDetector."""
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(edges)[0]
        if detected is None:
            return []

        height, width = edges.shape[:2]
        horizontal = np.zeros_like(edges)
        vertical = np.zeros_like(edges)

        for x1, y1, x2, y2 in np.rint(detected.reshape(-1, 4)).astype(np.int32):
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda p: p[0])
                cv2.line(horizontal, (max(x1 - 5, 0), y1), (min(x2 + 5, width - 1), y2), 255, 2)
            else:
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda p: p[1])
                cv2.line(vertical, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, height - 1)), 255, 2)

        corners: list[tuple[int, int]] = []
        final_horizontal = np.zeros_like(edges)
        final_vertical = np.zeros_like(edges)

        for canvas, final_canvas, axis in ((horizontal, final_horizontal, 0), (vertical, final_vertical, 1)):
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]:
                points = contour.reshape(-1, 2)
                low, high = points[:, axis].min() + 2, points[:, axis].max() - 2
                if high <= low:
                    continue
                if axis == 0:
                    y1 = int(np.mean(points[points[:, 0] == low][:, 1]))
                    y2 = int(np.mean(points[points[:, 0] == high][:, 1]))
                    start, end = (int(low), y1), (int(high), y2)
                else:
                    x1 = int(np.mean(points[points[:, 1] == low][:, 0]))
                    x2 = int(np.mean(points[points[:, 1] == high][:, 0]))
                    start, end = (x1, int(low)), (x2, int(high))
                cv2.line(final_canvas, start, end, 1, 1)
                corners.extend((start, end))

        ys, xs = np.where((final_horizontal + final_vertical) == 2)
        corners.extend(zip(xs.tolist(), ys.tolist()))
        return self.filter_corners(corners)[:12]

    def is_valid_contour(self, contour: np.ndarray, width: int, height: int) -> bool:
        contour = np.asarray(contour).reshape(-1, 2)
        if len(contour) != 4 or not cv2.isContourConvex(contour.reshape(-1, 1, 2).astype(np.float32)):
            return False
        return (cv2.contourArea(contour.astype(np.float32)) > width * height * self.min_quad_area_ratio
                and self.angle_range(contour) < self.max_quad_angle_range)

    def get_contour(self, image: np.ndarray) -> np.ndarray:
        """Return document corners, or the image boundary when none is detected."""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        edges = cv2.Canny(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel), 0, 84)
        candidates: list[np.ndarray] = []

        corners = self.get_corners(edges)
        if len(corners) >= 4:
            for points in itertools.combinations(corners, 4):
                candidate = transform.order_points(np.asarray(points, dtype=np.float32))
                if self.is_valid_contour(candidate, width, height):
                    candidates.append(candidate)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            candidate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True).reshape(-1, 2)
            if self.is_valid_contour(candidate, width, height):
                candidates.append(candidate.astype(np.float32))

        if candidates:
            return max(candidates, key=cv2.contourArea).reshape(4, 2)
        return np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype=np.float32)

    def scan(self, image_path: Path, output_dir: Path) -> Path:
        """Scan a single document image file and save result to output directory."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original = image.copy()
        scale = image.shape[0] / 500.0
        preview = imutils.resize(image, height=500)
        contour = self.get_contour(preview)

        warped = transform.four_point_transform(original, contour * scale)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        scanned = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 15)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), scanned):
            raise OSError(f"Could not write output image: {output_path}")

        print(f"Processed {image_path.name} -> {output_path}")
        return output_path
