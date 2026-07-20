"""Production-Grade Document Image Enhancement Engine.

Deterministic Computer Vision pipeline for document background whitening, shadow removal,
adaptive contrast enhancement, noise reduction, and crisp text/signature preservation.
Matches quality standards of Adobe Scan, Microsoft Lens, and CamScanner Premium.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class QualityMetrics:
    """Quantitative image quality metrics calculated by QualityAnalyzer."""
    background_whiteness: float
    text_contrast: float
    noise_level: float
    shadow_gradient: float
    edge_sharpness: float
    processing_time_ms: float = 0.0


class QualityAnalyzer:
    """Internal analyzer measuring document lighting, contrast, noise, and text clarity."""

    @staticmethod
    def analyze(image: np.ndarray) -> QualityMetrics:
        start_time = time.perf_counter()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        scale = 400.0 / max(h, w)
        if scale < 1.0:
            preview = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            preview = gray

        bg_whiteness = float(np.percentile(preview, 95))
        p5 = float(np.percentile(preview, 5))
        text_contrast = float(bg_whiteness - p5)

        blurred = cv2.GaussianBlur(preview, (5, 5), 0)
        high_freq = cv2.absdiff(preview, blurred)
        noise_level = float(np.mean(high_freq))

        qh, qw = preview.shape[0] // 2, preview.shape[1] // 2
        quad_means = [
            np.mean(preview[:qh, :qw]),
            np.mean(preview[:qh, qw:]),
            np.mean(preview[qh:, :qw]),
            np.mean(preview[qh:, qw:]),
        ]
        shadow_gradient = float(np.std(quad_means))

        laplacian = cv2.Laplacian(preview, cv2.CV_64F)
        edge_sharpness = float(laplacian.var())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return QualityMetrics(
            background_whiteness=bg_whiteness,
            text_contrast=text_contrast,
            noise_level=noise_level,
            shadow_gradient=shadow_gradient,
            edge_sharpness=edge_sharpness,
            processing_time_ms=elapsed_ms,
        )


class DocumentEnhancer:
    """Production-ready document enhancement engine supporting multiple specialized scan modes."""

    @staticmethod
    def flatten_illumination(image: np.ndarray, target_dim: int = 600) -> np.ndarray:
        """Fast multi-scale background illumination estimation & division-based shadow removal."""
        is_color = len(image.shape) == 3
        h, w = image.shape[:2]

        scale = float(target_dim) / max(h, w)
        if scale < 1.0:
            small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
            small_img = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small_img = image
            small_h, small_w = h, w

        kernel_size = max(15, min(small_h, small_w) // 16 * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        if is_color:
            bg_small = np.zeros_like(small_img)
            for c in range(3):
                dilated = cv2.morphologyEx(small_img[:, :, c], cv2.MORPH_DILATE, kernel)
                bg_small[:, :, c] = cv2.medianBlur(dilated, 11)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            dilated = cv2.morphologyEx(small_img, cv2.MORPH_DILATE, kernel)
            bg_small = cv2.medianBlur(dilated, 11)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

        bg_float = np.maximum(bg.astype(np.float32), 1.0)
        img_float = image.astype(np.float32)
        normalized = (img_float / bg_float) * 255.0
        return np.clip(normalized, 0, 255).astype(np.uint8)

    @staticmethod
    def normalize_background_tone(image: np.ndarray, black_percentile: float = 1.5,
                                  white_percentile: float = 97.5) -> np.ndarray:
        """Dynamic white/black point stretching with a smooth contrast transfer."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        black_pt = float(np.percentile(gray, black_percentile))
        white_pt = float(np.percentile(gray, white_percentile))

        if white_pt <= black_pt + 10:
            white_pt = min(255.0, black_pt + 50.0)

        norm = (image.astype(np.float32) - black_pt) / (white_pt - black_pt)
        norm = np.clip(norm, 0.0, 1.0)

        curved = np.power(norm, 1.12) * 255.0
        return np.clip(curved, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_adaptive_clahe(image: np.ndarray, clip_limit: float = 2.0,
                             grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return clahe.apply(image)

    @staticmethod
    def reduce_noise(image: np.ndarray, strength: int = 5) -> np.ndarray:
        """Apply edge-preserving bilateral filtering."""
        if strength <= 0:
            return image
        return cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=25)

    @staticmethod
    def sharpen_edges(image: np.ndarray, amount: float = 0.4) -> np.ndarray:
        """Apply selective unsharp mask edge sharpening."""
        if amount <= 0.0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), 2.5)
        sharpened = cv2.addWeighted(image.astype(np.float32), 1.0 + amount,
                                   blurred.astype(np.float32), -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def sauvola_threshold(gray: np.ndarray, window_size: int = 25, k: float = 0.18, R: float = 128.0) -> np.ndarray:
        """Optimized Sauvola local adaptive thresholding."""
        if window_size % 2 == 0:
            window_size += 1

        half_w = window_size // 2
        padded = cv2.copyMakeBorder(gray, half_w, half_w, half_w, half_w, cv2.BORDER_REPLICATE)

        mean = cv2.boxFilter(padded, cv2.CV_64F, (window_size, window_size))[half_w:-half_w, half_w:-half_w]
        sqr_mean = cv2.boxFilter(padded.astype(np.float64) ** 2, cv2.CV_64F, (window_size, window_size))[half_w:-half_w, half_w:-half_w]

        variance = np.maximum(0.0, sqr_mean - (mean ** 2))
        std = np.sqrt(variance)

        threshold = mean * (1.0 + k * ((std / R) - 1.0))
        return np.where(gray > threshold, 255, 0).astype(np.uint8)

    @staticmethod
    def cleanup_borders(image: np.ndarray, border_pixels: int = 4) -> np.ndarray:
        """Clean edge border pixels around outer boundaries."""
        if border_pixels <= 0:
            return image

        result = image.copy()
        h, w = result.shape[:2]
        pad = min(border_pixels, h // 25, w // 25)
        if pad <= 0:
            return result

        fill_val = 255 if len(image.shape) == 2 else (255, 255, 255)
        result[:pad, :] = fill_val
        result[-pad:, :] = fill_val
        result[:, :pad] = fill_val
        result[:, -pad:] = fill_val
        return result

    @classmethod
    def enhance_black_white(cls, image: np.ndarray) -> np.ndarray:
        """Production B&W scan mode: Illumination flattening -> Sauvola threshold -> Speckle cleanup."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        flattened = cls.flatten_illumination(gray, target_dim=600)
        denoised = cls.reduce_noise(flattened, strength=3)
        binary = cls.sauvola_threshold(denoised, window_size=25, k=0.16, R=128.0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cls.cleanup_borders(cleaned, border_pixels=3)

    @classmethod
    def enhance_magic_color(cls, image: np.ndarray) -> np.ndarray:
        """Adobe Scan / CamScanner Style: Whiten background while boosting ink & stamp color vibrancy."""
        if len(image.shape) == 2:
            return cls.enhance_black_white(image)

        flattened = cls.flatten_illumination(image, target_dim=600)
        lab = cv2.cvtColor(flattened, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        l_norm = cls.normalize_background_tone(l_chan, black_percentile=1.2, white_percentile=97.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_norm)

        lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
        bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_boosted = cv2.multiply(s, 1.3)
        hsv_boosted = cv2.merge([h, np.clip(s_boosted, 0, 255).astype(np.uint8), v])
        boosted_bgr = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

        sharpened = cls.sharpen_edges(boosted_bgr, amount=0.35)
        return cls.cleanup_borders(sharpened, border_pixels=3)

    @classmethod
    def enhance_color(cls, image: np.ndarray) -> np.ndarray:
        """Natural color enhancement: White balance correction and background illumination cleanup."""
        if len(image.shape) == 2:
            return image

        flattened = cls.flatten_illumination(image, target_dim=600)
        normalized = cls.normalize_background_tone(flattened, black_percentile=1.0, white_percentile=98.5)
        sharpened = cls.sharpen_edges(normalized, amount=0.3)
        return cls.cleanup_borders(sharpened, border_pixels=3)

    @classmethod
    def enhance_grayscale(cls, image: np.ndarray) -> np.ndarray:
        """Smooth shadow-free grayscale document scan."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        flattened = cls.flatten_illumination(gray, target_dim=600)
        normalized = cls.normalize_background_tone(flattened, black_percentile=1.2, white_percentile=98.0)
        sharpened = cls.sharpen_edges(normalized, amount=0.35)
        return cls.cleanup_borders(sharpened, border_pixels=3)

    @classmethod
    def enhance_high_contrast(cls, image: np.ndarray) -> np.ndarray:
        """High contrast document scan for faded text, pencil writing, or old receipts."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        flattened = cls.flatten_illumination(gray, target_dim=600)
        clahe_img = cls.apply_adaptive_clahe(flattened, clip_limit=3.0, grid_size=(8, 8))
        normalized = cls.normalize_background_tone(clahe_img, black_percentile=2.5, white_percentile=96.5)
        sharpened = cls.sharpen_edges(normalized, amount=0.5)
        return cls.cleanup_borders(sharpened, border_pixels=3)

    @classmethod
    def enhance_auto(cls, image: np.ndarray) -> np.ndarray:
        """Intelligent Auto mode: Analyzes image metrics and selects optimal processing parameters."""
        metrics = QualityAnalyzer.analyze(image)

        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))
            if sat_mean > 18.0:
                return cls.enhance_magic_color(image)

        if metrics.shadow_gradient > 12.0 or metrics.text_contrast < 80.0:
            return cls.enhance_black_white(image)

        if len(image.shape) == 3:
            return cls.enhance_magic_color(image)
        return cls.enhance_grayscale(image)

    @classmethod
    def process(cls, image: np.ndarray, mode: str) -> np.ndarray:
        """Main entrypoint for document enhancement."""
        mode_lower = mode.lower()
        if mode_lower == "original":
            return image
        if mode_lower == "auto":
            return cls.enhance_auto(image)
        if mode_lower == "magic_color":
            return cls.enhance_magic_color(image)
        if mode_lower in {"black_white", "black_and_white"}:
            return cls.enhance_black_white(image)
        if mode_lower == "color":
            return cls.enhance_color(image)
        if mode_lower == "grayscale":
            return cls.enhance_grayscale(image)
        if mode_lower == "high_contrast":
            return cls.enhance_high_contrast(image)

        raise ValueError(f"Unknown enhancement mode: {mode}")
