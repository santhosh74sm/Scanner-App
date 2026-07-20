"""Image manipulation helper functions."""
from __future__ import annotations

import cv2
import numpy as np


def resize(image: np.ndarray, width: int | None = None, height: int | None = None,
           inter: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize an image while preserving its original aspect ratio."""
    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (max(1, int(w * r)), height)
    else:
        r = width / float(w)
        dim = (width, max(1, int(h * r)))

    return cv2.resize(image, dim, interpolation=inter)
