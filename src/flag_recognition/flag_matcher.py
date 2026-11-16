"""Reference matching for clean national-flag pictures.

This is a deterministic supplement to the trained classifier, not a claim that
photo-scene recognition has been solved. References are independent public flag
artwork and are loaded with a bounded timeout.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import requests

REFERENCE_SIZE = (120, 80)
FLAGCDN_BASE = "https://flagcdn.com/w160"


@dataclass(frozen=True)
class FlagMatch:
    code: str | None
    distance: float | None
    margin: float | None
    matches: tuple[tuple[str, float], ...]
    accepted: bool


def _prepare(image: Image.Image) -> np.ndarray:
    """Normalize orientation and size without mirroring flag designs."""
    frame = ImageOps.exif_transpose(image)
    frame = frame.convert("RGBA")
    background = Image.new("RGBA", frame.size, (255, 255, 255, 255))
    background.alpha_composite(frame)
    rgb = background.convert("RGB")
    return np.asarray(
        rgb.resize(REFERENCE_SIZE, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def download_reference(code: str, timeout: float = 5.0) -> np.ndarray | None:
    """Fetch a canonical reference image; a failure must not block inference."""
    code = code.strip().lower()
    if len(code) != 2 or not code.isalpha():
        return None
    try:
        response = requests.get(
            f"{FLAGCDN_BASE}/{code}.png",
            timeout=timeout,
            headers={"User-Agent": "FlagIntelligence/0.2 (flag reference matching)"},
        )
        response.raise_for_status()
        if len(response.content) > 1_000_000:
            return None
        with Image.open(BytesIO(response.content)) as image:
            image.load()
            return _prepare(image)
    except (requests.RequestException, OSError, ValueError):
        return None


def build_gallery(codes: tuple[str, ...], max_workers: int = 12) -> dict[str, np.ndarray]:
    """Retrieve supported reference flags concurrently; allow partial results."""
    clean_codes = sorted({
        code.lower() for code in codes
        if isinstance(code, str) and len(code) == 2 and code.isalpha()
    })
    gallery: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_reference, code): code
            for code in clean_codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                image = future.result()
            except Exception:
                image = None
            if image is not None:
                gallery[code] = image
    return gallery


def rank_reference_flags(
    image: Image.Image,
    gallery: dict[str, np.ndarray],
    *,
    top_k: int = 5,
) -> FlagMatch:
    """Only accept a distinctive, close template match.

    The score is a normalized image difference, NOT a learned probability.
    Closely related and nearly identical flags remain ambiguous.
    """
    if not gallery or min(image.size) < 32:
        return FlagMatch(None, None, None, (), False)

    query = _prepare(image).astype(np.int16)
    distances = []
    for code, reference in gallery.items():
        if reference.shape != query.shape:
            continue
        difference = np.abs(query - reference.astype(np.int16))
        distance = float(difference.mean() / 255.0)
        distances.append((code, distance))
    if not distances:
        return FlagMatch(None, None, None, (), False)

    distances.sort(key=lambda x: x[1])
    best_code, best_distance = distances[0]
    margin = (
        distances[1][1] - best_distance
        if len(distances) > 1
        else 0.0
    )
    aspect = image.width / max(image.height, 1)
    accepted = (
        len(gallery) >= 100
        and 1.1 <= aspect <= 2.5
        and best_distance <= 0.105
        and margin >= 0.018
    )

    return FlagMatch(
        code=best_code,
        distance=best_distance,
        margin=margin,
        matches=tuple(distances[:max(1, top_k)]),
        accepted=accepted,
    )
