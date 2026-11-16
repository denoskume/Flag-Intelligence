"""High-precision recognition of simple vertical tricolor flags.

A deliberately narrow rule-based complement to the neural network. It handles
real cloth flags photographed on white backgrounds, for which a pixel-for-pixel
reference comparison is fragile. It never claims a probability or tries to
recognize all national flags.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class TricolorMatch:
    code: str
    strength: float
    candidates: tuple[tuple[str, float], ...]


# Restrict automatic acceptance to clear, non-ambiguous patterns.
# E.g. Romania and Chad are intentionally omitted because their shades are
# too similar for a guaranteed color-only decision.
PATTERNS: dict[str, tuple[str, str, str]] = {
    "fr": ("blue", "white", "red"),
    "ci": ("orange", "white", "green"),
    "ie": ("green", "white", "orange"),
}


def _color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Conservative color labels tolerant of folds, shadows and JPEG noise."""
    r, g, b = (rgb[..., index].astype(np.int16) for index in range(3))
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    return {
        "blue": (
            (b > 60) & (b > r * 1.30) & (b > g * 1.17)
            & ((b - r) > 25) & ((b - g) > 20)
        ),
        "white": (low > 155) & ((high - low) < 47),
        "red": (
            (r > 103) & (r > g * 1.42) & (r > b * 1.32)
            & ((r - g) > 60)
        ),
        "green": (
            (g > 65) & (g > r * 1.22) & (g > b * 1.13)
            & ((g - r) > 23)
        ),
        "orange": (
            (r > 120) & (g > 50) & (g < r * 0.83)
            & (b < g * 0.70) & ((r - g) > 30)
        ),
    }


def detect_vertical_tricolor(image: Image.Image) -> TricolorMatch | None:
    """Return a country only when wide, coherent vertical bands are visible.

    The flag crop is inferred from *colored* pixels (the middle stripe may be
    white) so small white margins, fabric creases and eyelets do not cause the
    canonical image-template score to fail. This path deliberately rejects
    screenshots, portrait scenes, horizontal flags and weak/ambiguous colors.
    """
    frame = ImageOps.exif_transpose(image).convert("RGB")
    width, height = frame.size
    if min(width, height) < 72 or not (1.12 <= width / height <= 3.15):
        return None

    frame.thumbnail((640, 480), Image.Resampling.BILINEAR)
    pixels = np.asarray(frame, dtype=np.uint8)
    maximum = pixels.max(axis=2).astype(np.int16)
    minimum = pixels.min(axis=2).astype(np.int16)
    saturated = ((maximum - minimum) > 50) & (maximum > 64)

    yy, xx = np.where(saturated)
    if len(xx) < 0.07 * pixels.shape[0] * pixels.shape[1]:
        return None

    x0, x1 = np.percentile(xx, [1, 99])
    y0, y1 = np.percentile(yy, [1, 99])
    cropped_width, cropped_height = x1 - x0, y1 - y0
    if (
        cropped_width < 80
        or cropped_height < 48
        or not 1.20 <= cropped_width / cropped_height <= 2.85
    ):
        return None

    crop = pixels[
        max(0, int(y0)):min(pixels.shape[0], int(y1) + 1),
        max(0, int(x0)):min(pixels.shape[1], int(x1) + 1),
    ]
    h, w = crop.shape[:2]
    # Sample the center of each vertical third across most of the flag height,
    # not one row; a horizontal tricolor cannot pass this coherence check.
    stripes = [
        crop[int(h * 0.18):int(h * 0.82), int(w * x_low):int(w * x_high)]
        for x_low, x_high in ((0.12, 0.28), (0.42, 0.58), (0.72, 0.88))
    ]
    if any(stripe.size == 0 for stripe in stripes):
        return None

    strengths = [
        {name: float(mask.mean()) for name, mask in _color_masks(stripe).items()}
        for stripe in stripes
    ]
    ranking = []
    for code, expected_colors in PATTERNS.items():
        coverage = [strengths[index][color] for index, color in enumerate(expected_colors)]
        ranking.append((code, min(coverage)))
    ranking.sort(key=lambda item: item[1], reverse=True)

    code, best = ranking[0]
    runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
    # Not a probability: this is the minimum correctly-colored pixel fraction
    # in the three expected regions.
    if best < 0.84 or best - runner_up < 0.30:
        return None
    return TricolorMatch(
        code=code,
        strength=best,
        candidates=tuple(ranking),
    )
