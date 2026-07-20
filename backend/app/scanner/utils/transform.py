"""Perspective transformation and point ordering utilities using OpenCV and NumPy."""
from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray | list[list[float]]) -> np.ndarray:
    """Order four 2D points in top-left, top-right, bottom-right, bottom-left order."""
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x_sorted = pts_arr[np.argsort(pts_arr[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most[0], left_most[1]

    distances = np.linalg.norm(right_most - tl, axis=1)
    sorted_right = right_most[np.argsort(distances)[::-1], :]
    br, tr = sorted_right[0], sorted_right[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray | list[list[float]]) -> np.ndarray:
    """Apply a 4-point perspective transform to extract a flattened birds-eye view."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(1, int(round(max(width_a, width_b))))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(1, int(round(max(height_a, height_b))))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
