"""Production-Grade Document Image Enhancement Engine with Memory Optimization for 512MB RAM environments.

Clean Computer Vision pipeline for B&W Clean document enhancement:
1. Preprocessing:
   Automatic White Balance -> Multi-scale Illumination Division -> Adaptive Paper Whitening.
2. Detail-Preserving B&W Clean Rendering:
   Detail-preserving monochrome thresholding (pure black text, pure white paper).
"""
from __future__ import annotations

import gc
import cv2
import numpy as np


class DocumentEnhancer:
    """Production-ready document enhancement engine optimized for low-memory runtimes."""

    MAX_ENHANCE_DIM = 2200  # Cap maximum dimension for heavy matrix calculations (~300 DPI output)

    @staticmethod
    def analyze_scene(image: np.ndarray) -> dict[str, float]:
        """Analyze scene metrics on a small thumbnail to minimize RAM allocation."""
        h, w = image.shape[:2]
        scale = 500.0 / max(h, w)
        if scale < 1.0:
            thumb = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            thumb = image

        if len(thumb.shape) == 3:
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            # Saturation estimate without allocating full HSV matrix
            max_c = np.max(thumb, axis=2)
            min_c = np.min(thumb, axis=2)
            sat_mean = float(np.mean(max_c - min_c))
            del max_c, min_c
        else:
            gray = thumb
            sat_mean = 0.0

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        p10 = float(np.percentile(gray, 10))
        p90 = float(np.percentile(gray, 90))
        shadow_intensity = max(0.0, (p90 - p10) / (p90 + 1e-5))

        if scale < 1.0:
            del thumb, gray

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

        p98_b = float(np.percentile(image[:, :, 0], 98))
        p98_g = float(np.percentile(image[:, :, 1], 98))
        p98_r = float(np.percentile(image[:, :, 2], 98))

        max_val = max(p98_b, p98_g, p98_r, 1.0)
        scale_b = float(np.clip(max_val / max(p98_b, 1.0), 0.85, 1.35))
        scale_g = float(np.clip(max_val / max(p98_g, 1.0), 0.85, 1.35))
        scale_r = float(np.clip(max_val / max(p98_r, 1.0), 0.85, 1.35))

        result = image.astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r

        out = np.clip(result, 0, 255).astype(np.uint8)
        del result
        return out

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
                del dilated
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
            del bg_small
        else:
            dilated = cv2.morphologyEx(small_img, cv2.MORPH_DILATE, kernel)
            bg_small = cv2.GaussianBlur(dilated.astype(np.float32), (31, 31), 0)
            del dilated
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
            del bg_small

        if scale < 1.0:
            del small_img

        bg_float = np.maximum(bg, 1.0)
        del bg
        img_float = image.astype(np.float32)
        normalized = (img_float / bg_float) * 240.0
        del bg_float, img_float

        out = np.clip(normalized, 0, 255).astype(np.uint8)
        del normalized
        return out

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
        black_thresh = max(0.0, min(paper_thresh - 30.0, p3 + 10.0))

        img_float = image.astype(np.float32)
        denom = max(1.0, paper_thresh - black_thresh)

        if len(image.shape) == 3:
            whitened = np.zeros_like(img_float)
            for c in range(3):
                chan = img_float[:, :, c]
                norm = np.clip((chan - black_thresh) / denom, 0.0, 1.0) * 255.0
                whitened[:, :, c] = np.where(chan >= paper_thresh, target_paper_min, norm)
                del chan, norm
        else:
            norm = np.clip((img_float - black_thresh) / denom, 0.0, 1.0) * 255.0
            whitened = np.where(img_float >= paper_thresh, target_paper_min, norm)
            del norm

        del img_float
        if len(image.shape) == 3 or image is not gray:
            del gray

        out = np.clip(whitened, 0, 255).astype(np.uint8)
        del whitened
        return out

    @classmethod
    def preprocess_paper_background(cls, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Execute White Balance -> Illumination Division -> Paper Whitening with intermediate cleanup."""
        wb_img = cls.automatic_white_balance(image)
        flattened = cls.flatten_illumination(wb_img, target_dim=350)
        preprocessed = cls.adaptive_paper_whitening(flattened, target_paper_min=255.0)
        del flattened
        return preprocessed, wb_img

    @staticmethod
    def sharpen_text(image: np.ndarray, amount: float = 0.35) -> np.ndarray:
        """Apply unsharp mask edge sharpening using float32 memory precision."""
        if amount <= 0.0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), 2.5)
        img_f = image.astype(np.float32)
        blur_f = blurred.astype(np.float32)
        del blurred
        sharpened = cv2.addWeighted(img_f, 1.0 + amount, blur_f, -amount, 0)
        del img_f, blur_f
        out = np.clip(sharpened, 0, 255).astype(np.uint8)
        del sharpened
        return out

    @staticmethod
    def sauvola_threshold(gray: np.ndarray, window_size: int = 25, k: float = 0.12, R: float = 128.0) -> np.ndarray:
        """Sauvola local adaptive threshold using 32-bit float precision to reduce memory by 50%."""
        if window_size % 2 == 0:
            window_size += 1

        half_w = window_size // 2
        padded = cv2.copyMakeBorder(gray, half_w, half_w, half_w, half_w, cv2.BORDER_REPLICATE)

        mean = cv2.boxFilter(padded, cv2.CV_32F, (window_size, window_size))[half_w:-half_w, half_w:-half_w]
        padded_f32 = padded.astype(np.float32)
        del padded
        sqr = padded_f32 * padded_f32
        del padded_f32

        sqr_mean = cv2.boxFilter(sqr, cv2.CV_32F, (window_size, window_size))[half_w:-half_w, half_w:-half_w]
        del sqr

        variance = np.maximum(0.0, sqr_mean - (mean * mean))
        del sqr_mean
        std = np.sqrt(variance)
        del variance

        threshold = mean * (1.0 + k * ((std / R) - 1.0))
        del mean, std
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
        """B&W Clean: Detail-preserving binarization with dimension capping for < 100MB RAM footprint."""
        orig_h, orig_w = image.shape[:2]

        # Downscale max dimension if exceeding MAX_ENHANCE_DIM to bound matrix RAM allocations
        if max(orig_h, orig_w) > cls.MAX_ENHANCE_DIM:
            scale = cls.MAX_ENHANCE_DIM / float(max(orig_h, orig_w))
            work_h, work_w = max(1, int(orig_h * scale)), max(1, int(orig_w * scale))
            proc_img = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            proc_img = image

        if len(proc_img.shape) == 3:
            gray_input = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_input = proc_img.copy()

        scene = cls.analyze_scene(gray_input)

        # UNIFIED PREPROCESSING PIPELINE
        preprocessed, _ = cls.preprocess_paper_background(proc_img)
        if proc_img is not image:
            del proc_img

        if len(preprocessed.shape) == 3:
            gray_prep = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
            del preprocessed
        else:
            gray_prep = preprocessed

        sauvola_k = 0.12 if scene["contrast"] > 25 else 0.10
        threshold_surface = cls.sauvola_threshold(gray_prep, window_size=25, k=sauvola_k, R=128.0)

        # Detail-preserving soft adaptive thresholding
        margin = 20.0
        gray_f32 = gray_prep.astype(np.float32)
        bw_float = np.clip((gray_f32 - (threshold_surface - margin)) / (2.0 * margin), 0.0, 1.0) * 255.0

        # Paper background becomes pure 255 white
        final_bw = np.where(gray_prep >= (threshold_surface + margin), 255.0, bw_float)
        # Deep text stroke becomes pure 0 black
        final_bw = np.where(gray_prep <= (threshold_surface - margin), 0.0, final_bw)
        del bw_float, threshold_surface
        # Paper background pixels cleanly mapped to 255
        final_bw = np.where(gray_prep >= 250.0, 255.0, final_bw)
        del gray_prep, gray_f32, gray_input

        final_bw = np.clip(final_bw, 0, 255).astype(np.uint8)

        sharpen_amount = 0.25 if scene["noise"] < 500 else 0.10
        sharpened = cls.sharpen_text(final_bw, amount=sharpen_amount)
        del final_bw

        cleaned = cls.cleanup_borders(sharpened, border_pixels=3)
        del sharpened

        gc.collect()
        return cleaned

    @classmethod
    def process(cls, image: np.ndarray, mode: str = "black_white") -> np.ndarray:
        """Main entrypoint routing to B&W Clean filter."""
        return cls.enhance_black_white(image)
