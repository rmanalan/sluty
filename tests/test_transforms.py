import math
import numpy as np
import pytest

from npc_to_lut.transforms import (
    _rgb_to_hsl,
    _hsl_to_rgb,
    apply_tone_curve,
    apply_contrast,
    apply_highlights,
    apply_shadows,
    apply_saturation,
    apply_color_blender,
    apply_white_black,
)
from npc_to_lut.parser import ColorBlenderBand


# ---------------------------------------------------------------------------
# HSL round-trip
# ---------------------------------------------------------------------------

def test_hsl_round_trip():
    rng = np.random.default_rng(42)
    rgb = rng.random((500, 3)).astype(np.float64)
    recovered = _hsl_to_rgb(_rgb_to_hsl(rgb))
    np.testing.assert_allclose(rgb, recovered, atol=1e-6)


def test_pure_red_hsl():
    red = np.array([[1.0, 0.0, 0.0]])
    hsl = _rgb_to_hsl(red)
    assert abs(hsl[0, 0]) < 1.0    # hue ≈ 0°
    assert abs(hsl[0, 1] - 1.0) < 1e-6  # saturation = 1
    assert abs(hsl[0, 2] - 0.5) < 1e-6  # lightness = 0.5


# ---------------------------------------------------------------------------
# Tone curve
# ---------------------------------------------------------------------------

def test_tone_curve_identity():
    lut = list(np.linspace(0, 1, 257))
    rgb = np.array([[0.2, 0.4, 0.6], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = apply_tone_curve(rgb, lut)
    np.testing.assert_allclose(result, rgb, atol=1e-4)


def test_tone_curve_invert():
    """Inverted curve: every output = 1 - input on luma."""
    lut = list(np.linspace(1, 0, 257))   # reversed
    gray = np.array([[0.5, 0.5, 0.5]])
    result = apply_tone_curve(gray, lut)
    # Midgray (0.5) maps to ~0.5 under inverted curve → luma_out ≈ 0.5
    assert 0.4 < result[0, 0] < 0.6


def test_tone_curve_none_is_identity():
    rgb = np.array([[0.3, 0.5, 0.7]])
    result = apply_tone_curve(rgb, None)
    np.testing.assert_array_equal(result, rgb)


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

def test_contrast_zero_is_identity():
    rgb = np.random.default_rng(1).random((100, 3))
    result = apply_contrast(rgb, 0.0)
    np.testing.assert_array_equal(result, rgb)


def test_contrast_positive_stretches_midtones():
    """After positive contrast, values above 0.5 should get brighter, below darker."""
    above = np.array([[0.75, 0.75, 0.75]])
    below = np.array([[0.25, 0.25, 0.25]])
    above_out = apply_contrast(above, 50.0)
    below_out = apply_contrast(below, 50.0)
    assert above_out[0, 0] > 0.75
    assert below_out[0, 0] < 0.25


def test_contrast_midgray_unchanged():
    gray = np.array([[0.5, 0.5, 0.5]])
    result = apply_contrast(gray, 80.0)
    np.testing.assert_allclose(result, gray, atol=1e-6)


# ---------------------------------------------------------------------------
# Highlights / Shadows
# ---------------------------------------------------------------------------

def test_highlights_dark_pixels_unaffected():
    dark = np.array([[0.1, 0.1, 0.1]])
    result = apply_highlights(dark, 50.0)
    np.testing.assert_allclose(result, dark, atol=0.02)


def test_highlights_bright_pixels_boosted():
    bright = np.array([[0.9, 0.9, 0.9]])
    result = apply_highlights(bright, 50.0)
    assert result[0, 0] > 0.9


def test_shadows_bright_pixels_unaffected():
    bright = np.array([[0.9, 0.9, 0.9]])
    result = apply_shadows(bright, 50.0)
    np.testing.assert_allclose(result, bright, atol=0.02)


def test_shadows_dark_pixels_lifted():
    dark = np.array([[0.05, 0.05, 0.05]])
    result = apply_shadows(dark, 50.0)
    assert result[0, 0] > 0.05


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def test_saturation_zero_is_identity():
    rgb = np.random.default_rng(2).random((100, 3))
    result = apply_saturation(rgb, 0.0)
    np.testing.assert_array_equal(result, rgb)


def test_saturation_minus100_is_grayscale():
    """At -100, all colours should become gray (S=0)."""
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.2, 0.8]])
    result = apply_saturation(rgb, -100.0)
    for row in result:
        assert abs(row[0] - row[1]) < 1e-6
        assert abs(row[1] - row[2]) < 1e-6


def test_saturation_gray_unchanged():
    """Neutral gray has no chroma; saturation changes nothing."""
    gray = np.array([[0.5, 0.5, 0.5]])
    result = apply_saturation(gray, 50.0)
    np.testing.assert_allclose(result, gray, atol=1e-6)


# ---------------------------------------------------------------------------
# Color Blender
# ---------------------------------------------------------------------------

def _neutral_bands():
    return [ColorBlenderBand(0, 0, 0) for _ in range(8)]


def test_color_blender_neutral_is_identity():
    rgb = np.random.default_rng(3).random((200, 3))
    result = apply_color_blender(rgb, _neutral_bands())
    np.testing.assert_array_equal(result, rgb)


def test_color_blender_red_hue_shift():
    """A +60° hue shift on the Red band should rotate pure red toward orange/yellow."""
    bands = _neutral_bands()
    bands[0] = ColorBlenderBand(hue=60, chroma=0, brightness=0)
    red = np.array([[1.0, 0.0, 0.0]])
    result = apply_color_blender(red, bands)
    hsl_in  = _rgb_to_hsl(red)
    hsl_out = _rgb_to_hsl(result)
    # Hue should increase (shift toward orange)
    assert hsl_out[0, 0] > hsl_in[0, 0]


# ---------------------------------------------------------------------------
# White / Black level
# ---------------------------------------------------------------------------

def test_white_black_zero_identity():
    rgb = np.random.default_rng(4).random((100, 3))
    result = apply_white_black(rgb, 0.0, 0.0)
    np.testing.assert_array_equal(result, rgb)
