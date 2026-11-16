"""Regression tests for the real-world tricolor failure (France screenshot).

The user screenshot is deliberately NOT committed to a public repository.
These synthetic pictures reproduce its dark navy, fabric texture and margins.
"""
import numpy as np
from PIL import Image, ImageDraw

from flag_recognition.tricolor import detect_vertical_tricolor


def _fabric_flag(left, center, right, *, padding=22, seed=4):
    width, height = 480, 300
    canvas = np.full((height + 2 * padding, width + 2 * padding, 3), 250, dtype=np.uint8)
    bands = np.zeros((height, width, 3), dtype=np.float32)
    for x0, x1, color in ((0, 160, left), (160, 320, center), (320, 480, right)):
        bands[:, x0:x1, :] = color
    rng = np.random.default_rng(seed)
    shade = 0.91 + 0.06 * np.sin(np.arange(width)[None, :] / 15) + 0.03 * np.sin(np.arange(height)[:, None] / 20)
    noise = rng.normal(0, 3, size=bands.shape)
    bands = np.clip(bands * shade[:, :, None] + noise, 0, 255).astype(np.uint8)
    canvas[padding:padding + height, padding:padding + width] = bands
    return Image.fromarray(canvas)


def test_real_world_dark_navy_french_fabric_and_margin_is_identified():
    # Dark navy and red represent the uploaded example that the original app rejected.
    photo = _fabric_flag((12, 22, 94), (250, 250, 250), (215, 4, 18))
    match = detect_vertical_tricolor(photo)
    assert match is not None
    assert match.code == "fr"
    assert match.strength >= 0.84


def test_reverse_orange_white_green_orders_remain_distinct():
    ivory_coast = _fabric_flag((246, 128, 0), (247, 247, 247), (0, 150, 88))
    ireland = _fabric_flag((19, 156, 95), (250, 250, 250), (255, 138, 62))
    assert detect_vertical_tricolor(ivory_coast).code == "ci"
    assert detect_vertical_tricolor(ireland).code == "ie"


def test_horizontal_french_colors_are_not_confused_with_vertical_france():
    image = Image.new("RGB", (480, 300))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 480, 99), fill=(10, 25, 100))
    draw.rectangle((0, 100, 480, 199), fill=(240, 240, 240))
    draw.rectangle((0, 200, 480, 300), fill=(220, 10, 20))
    assert detect_vertical_tricolor(image) is None


def test_unrelated_images_and_portrait_screenshots_are_rejected():
    image = Image.new("RGB", (480, 300), (220, 225, 232))
    assert detect_vertical_tricolor(image) is None
    portrait = Image.new("RGB", (360, 800), (250, 250, 250))
    assert detect_vertical_tricolor(portrait) is None


def test_no_forced_prediction_for_romanian_chadian_like_blue_yellow_red():
    # Both Romania and Chad require a more discriminating model than this rule.
    image = _fabric_flag((10, 30, 135), (247, 210, 0), (207, 19, 34))
    assert detect_vertical_tricolor(image) is None
