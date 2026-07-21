"""Production-Grade Document Image Enhancement Engine.

Clean Computer Vision pipeline for B&W Clean document enhancement:
1. Preprocessing:
   Automatic White Balance -> Multi-scale Illumination Division -> Adaptive Paper Whitening (Pure White Paper RGB 255).
2. Detail-Preserving B&W Clean Rendering:
   Detail-preserving monochrome thresholding (pure black text, pure white paper, preserved headers/lines/stamps/photos).
"""
from __future__ import annotations

import cv2
import numpy as np


class DocumentEnhancer:
    """Production-ready document enhancement engine supporting B&W Clean mode."""

    @staticmethod
    def analyze_scene(image: np.ndarray) -> dict[str, float]:
        """Automatically analyze scene metrics to adaptively tune enhancement parameters."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))
        else:
            gray = image
            sat_mean = 0.0

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        p10 = float(np.percentile(gray, 10))
        p90 = float(np.percentile(gray, 90))
        shadow_intensity = max(0.0, (p90 - p10) / (p90 + 1e-5))

        return {
            "brightness": brightness,
            "contrast": contrast,
            "noise": laplacian_var,
            "saturation": sat_mean,
            "shadow_intensity": shadow_intensity,
        }

    @staticmethod
    def automatic_white_balance(image: np.ndarray) -> np.ndarray:
        """Correct warm/cool camera color casts using percentile channel scaling."""
        if len(image.shape) == 2:
            return image

        result = image.astype(np.float32)
        p98_b = float(np.percentile(result[:, :, 0], 98))
        p98_g = float(np.percentile(result[:, :, 1], 98))
        p98_r = float(np.percentile(result[:, :, 2], 98))

        max_val = max(p98_b, p98_g, p98_r, 1.0)
        scale_b = max_val / max(p98_b, 1.0)
        scale_g = max_val / max(p98_g, 1.0)
        scale_r = max_val / max(p98_r, 1.0)

        result[:, :, 0] *= float(np.clip(scale_b, 0.85, 1.35))
        result[:, :, 1] *= float(np.clip(scale_g, 0.85, 1.35))
        result[:, :, 2] *= float(np.clip(scale_r, 0.85, 1.35))

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def flatten_illumination(image: np.ndarray, target_dim: int = 350) -> np.ndarray:
        """Multi-scale background illumination estimation & division-based shadow removal."""
        is_color = len(image.shape) == 3
        h, w = image.shape[:2]

        scale = float(target_dim) / max(h, w)
        if scale < 1.0:
            small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
            small_img = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small_img = image
            small_h, small_w = h, w

        kernel_size = max(21, min(small_h, small_w) // 8 * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        if is_color:
            bg_small = np.zeros_like(small_img, dtype=np.float32)
            for c in range(3):
                dilated = cv2.morphologyEx(small_img[:, :, c], cv2.MORPH_DILATE, kernel)
                bg_small[:, :, c] = cv2.GaussianBlur(dilated.astype(np.float32), (31, 31), 0)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            dilated = cv2.morphologyEx(small_img, cv2.MORPH_DILATE, kernel)
            bg_small = cv2.GaussianBlur(dilated.astype(np.float32), (31, 31), 0)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

        bg_float = np.maximum(bg, 1.0)
        img_float = image.astype(np.float32)
        normalized = (img_float / bg_float) * 240.0
        return np.clip(normalized, 0, 255).astype(np.uint8)

    @staticmethod
    def adaptive_paper_whitening(image: np.ndarray, target_paper_min: float = 255.0) -> np.ndarray:
        """Adaptive paper whitening mapping paper background pixels cleanly to pure white (RGB 255)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        p95 = float(np.percentile(gray, 95))
        p3 = float(np.percentile(gray, 3))

        paper_thresh = max(200.0, p95 - 15.0)
        black_thresh = min(paper_thresh - 30.0, p3 + 10.0)
        black_thresh = max(0.0, black_thresh)

        img_float = image.astype(np.float32)

        if len(image.shape) == 3:
            whitened = np.zeros_like(img_float)
            for c in range(3):
                chan = img_float[:, :, c]
                norm = (chan - black_thresh) / (paper_thresh - black_thresh)
                norm = np.clip(norm, 0.0, 1.0)
                whitened_chan = norm * 255.0
                whitened[:, :, c] = np.where(chan >= paper_thresh, target_paper_min, whitened_chan)
        else:
            norm = (img_float - black_thresh) / (paper_thresh - black_thresh)
            norm = np.clip(norm, 0.0, 1.0)
            whitened = norm * 255.0
            whitened = np.where(img_float >= paper_thresh, target_paper_min, whitened)

        return np.clip(whitened, 0, 255).astype(np.uint8)

    @classmethod
    def preprocess_paper_background(cls, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """UNIFIED COMMON PREPROCESSING:

        Executes White Balance -> Multi-scale Illumination Division -> Adaptive Paper Whitening.
        Returns tuple of (preprocessed_color_or_gray, white_balanced_original).
        """
        # Step 1: Automatic White Balance
        wb_img = cls.automatic_white_balance(image)

        # Step 2: Multi-Scale Illumination Division & Shadow Removal
        flattened = cls.flatten_illumination(wb_img, target_dim=350)

        # Step 3: Adaptive Paper Whitening (Forces paper background to RGB 255)
        preprocessed = cls.adaptive_paper_whitening(flattened, target_paper_min=255.0)

        return preprocessed, wb_img

    @staticmethod
    def sharpen_text(image: np.ndarray, amount: float = 0.35) -> np.ndarray:
        """Apply unsharp mask edge sharpening on text strokes."""
        if amount <= 0.0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), 2.5)
        sharpened = cv2.addWeighted(image.astype(np.float32), 1.0 + amount,
                                   blurred.astype(np.float32), -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def sauvola_threshold(gray: np.ndarray, window_size: int = 25, k: float = 0.12, R: float = 128.0) -> np.ndarray:
        """Sauvola local adaptive threshold surface computation."""
        if window_size % 2 == 0:
            window_size += 1

        half_w = window_size // 2
        padded = cv2.copyMakeBorder(gray, half_w, half_w, half_w, half_w, cv2.BORDER_REPLICATE)

        mean = cv2.boxFilter(padded, cv2.CV_64F, (window_size, window_size))[half_w:-half_w, half_w:-half_w]
        sqr_mean = cv2.boxFilter(padded.astype(np.float64) ** 2, cv2.CV_64F, (window_size, window_size))[half_w:-half_w, half_w:-half_w]

        variance = np.maximum(0.0, sqr_mean - (mean ** 2))
        std = np.sqrt(variance)

        threshold = mean * (1.0 + k * ((std / R) - 1.0))
        return threshold

    @staticmethod
    def cleanup_borders(image: np.ndarray, border_pixels: int = 3) -> np.ndarray:
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
        """B&W Clean: Detail-preserving binarization on top of UNIFIED PURE WHITE paper background."""
        if len(image.shape) == 3:
            gray_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_input = image.copy()

        scene = cls.analyze_scene(gray_input)

        # UNIFIED PREPROCESSING PIPELINE
        preprocessed, _ = cls.preprocess_paper_background(image)
        if len(preprocessed.shape) == 3:
            gray_prep = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        else:
            gray_prep = preprocessed

        sauvola_k = 0.12 if scene["contrast"] > 25 else 0.10
        threshold_surface = cls.sauvola_threshold(gray_prep, window_size=25, k=sauvola_k, R=128.0)

        # Detail-preserving soft adaptive thresholding
        margin = 20.0
        bw_float = (gray_prep.astype(np.float32) - (threshold_surface - margin)) / (2.0 * margin)
        bw_float = np.clip(bw_float, 0.0, 1.0) * 255.0

        # Paper background (above threshold + margin) becomes pure 255 white
        final_bw = np.where(gray_prep >= (threshold_surface + margin), 255.0, bw_float)
        # Deep text stroke (below threshold - margin) becomes pure 0 black
        final_bw = np.where(gray_prep <= (threshold_surface - margin), 0.0, final_bw)
        # Paper background pixels from preprocessed cleanly mapped to 255
        final_bw = np.where(gray_prep >= 250.0, 255.0, final_bw)

        final_bw = np.clip(final_bw, 0, 255).astype(np.uint8)

        sharpen_amount = 0.25 if scene["noise"] < 500 else 0.10
        sharpened = cls.sharpen_text(final_bw, amount=sharpen_amount)

        return cls.cleanup_borders(sharpened, border_pixels=3)

    @classmethod
    def process(cls, image: np.ndarray, mode: str = "black_white") -> np.ndarray:
        """Main entrypoint routing to B&W Clean filter."""
        return cls.enhance_black_white(image)
