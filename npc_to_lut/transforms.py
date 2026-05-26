"""
Color transform functions for LUT construction.

All functions accept and return numpy arrays shaped (N, 3) with values in [0, 1]
representing sRGB linear-ish image data (post-gamma, as stored in JPEGs).

Processing order expected by lut_builder:
  tone_curve → contrast → highlights/shadows → white/black → saturation
  → color_blender → color_grading
"""

import math
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers — RGB ↔ HSL
# ---------------------------------------------------------------------------

def _rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """Convert (N,3) RGB [0,1] to HSL [H∈0..360, S∈0..1, L∈0..1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    l = (cmax + cmin) / 2.0

    s = np.where(delta < 1e-9, 0.0, delta / (1.0 - np.abs(2 * l - 1) + 1e-9))
    s = np.clip(s, 0, 1)

    h = np.zeros(len(r))
    mask = delta > 1e-9

    # H when max is R
    m = mask & (cmax == r)
    h[m] = 60.0 * (((g[m] - b[m]) / delta[m]) % 6)

    # H when max is G
    m = mask & (cmax == g)
    h[m] = 60.0 * ((b[m] - r[m]) / delta[m] + 2)

    # H when max is B
    m = mask & (cmax == b)
    h[m] = 60.0 * ((r[m] - g[m]) / delta[m] + 4)

    h = h % 360.0

    return np.stack([h, s, l], axis=-1)


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Convert (N,3) HSL [H∈0..360, S∈0..1, L∈0..1] to RGB [0,1]."""
    h, s, l = hsl[:, 0], hsl[:, 1], hsl[:, 2]

    c = (1.0 - np.abs(2 * l - 1)) * s
    x = c * (1.0 - np.abs((h / 60.0) % 2 - 1))
    m = l - c / 2.0

    r = np.zeros(len(h))
    g = np.zeros(len(h))
    b = np.zeros(len(h))

    for lo, hi, ri, gi, bi in [
        (0,   60,  c, x, 0),
        (60,  120, x, c, 0),
        (120, 180, 0, c, x),
        (180, 240, 0, x, c),
        (240, 300, x, 0, c),
        (300, 360, c, 0, x),
    ]:
        mask = (h >= lo) & (h < hi)
        r[mask] = ri[mask] if isinstance(ri, np.ndarray) else ri
        g[mask] = gi[mask] if isinstance(gi, np.ndarray) else gi
        b[mask] = bi[mask] if isinstance(bi, np.ndarray) else bi

    return np.clip(np.stack([r + m, g + m, b + m], axis=-1), 0, 1)


# ---------------------------------------------------------------------------
# Luminance helper (sRGB coefficients, applied in gamma-encoded space)
# ---------------------------------------------------------------------------

_LUMA_COEFFS = np.array([0.2126, 0.7152, 0.0722])


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (rgb * _LUMA_COEFFS).sum(axis=-1)


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------

def apply_tone_curve(rgb: np.ndarray, lut: Optional[list]) -> np.ndarray:
    """
    Apply a 257-entry luminance tone curve.
    The curve maps input luminance in [0,1] to output luminance in [0,1].
    Chroma is preserved by scaling RGB channels proportionally.
    """
    if lut is None:
        return rgb

    lut_arr = np.array(lut, dtype=np.float64)   # shape (257,)
    t = np.linspace(0, 1, 257)

    luma_in = _luma(rgb)
    luma_out = np.interp(luma_in, t, lut_arr)

    # Avoid division by zero for very dark pixels
    safe_in = np.where(luma_in > 1e-6, luma_in, 1.0)
    ratio = np.where(luma_in > 1e-6, luma_out / safe_in, luma_out)
    return np.clip(rgb * ratio[:, np.newaxis], 0, 1)


def apply_contrast(rgb: np.ndarray, contrast: float) -> np.ndarray:
    """
    Parametric S-curve contrast around mid-gray (0.5).
    contrast ∈ [-100, 100]; 0 = identity.
    factor = 3^(contrast/100): gives 3x slope at +100, 1x at 0, 1/3x at -100.
    """
    if abs(contrast) < 1e-6:
        return rgb

    factor = math.exp(math.log(3.0) * contrast / 100.0)
    out = 0.5 + (rgb - 0.5) * factor
    return np.clip(out, 0, 1)


def apply_highlights(rgb: np.ndarray, highlights: float) -> np.ndarray:
    """Boost or roll-off highlights. highlights ∈ [-100, 100]."""
    if abs(highlights) < 1e-6:
        return rgb

    luma = _luma(rgb)
    # Mask rises from 0 at luma=0.5 to 1 at luma=1.0
    mask = np.clip(2 * luma - 1.0, 0, 1) ** 2
    delta = mask * (highlights / 100.0) * 0.5   # max ±0.5 shift
    return np.clip(rgb + delta[:, np.newaxis], 0, 1)


def apply_shadows(rgb: np.ndarray, shadows: float) -> np.ndarray:
    """Lift or crush shadows. shadows ∈ [-100, 100]."""
    if abs(shadows) < 1e-6:
        return rgb

    luma = _luma(rgb)
    mask = np.clip(1.0 - 2 * luma, 0, 1) ** 2
    delta = mask * (shadows / 100.0) * 0.5
    return np.clip(rgb + delta[:, np.newaxis], 0, 1)


def apply_white_black(rgb: np.ndarray, white_level: float, black_level: float) -> np.ndarray:
    """
    Output-range adjustment.
    white_level > 0 rolls off whites; black_level > 0 lifts blacks.
    Each unit = 0.1% of full range (so ±100 = ±10% of the output range).
    """
    if abs(white_level) < 1e-6 and abs(black_level) < 1e-6:
        return rgb

    black_out = np.clip(black_level / 1000.0, -0.5, 0.5)
    white_out = np.clip(1.0 - white_level / 1000.0, 0.5, 1.5)
    return np.clip(black_out + rgb * (white_out - black_out), 0, 1)


def apply_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    """Scale HSL chroma. saturation ∈ [-100, 100]; 0 = identity."""
    if abs(saturation) < 1e-6:
        return rgb

    hsl = _rgb_to_hsl(rgb)
    hsl[:, 1] *= max(0.0, 1.0 + saturation / 100.0)
    hsl[:, 1]  = np.clip(hsl[:, 1], 0, 1)
    return _hsl_to_rgb(hsl)


# Hue centres for the 8 Color Blender bands (degrees):
# Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta
_CB_HUE_CENTERS = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0])
_CB_HALF_WIDTH  = 45.0   # Gaussian half-width in degrees (approximation)


def _hue_weight(pixel_hue: np.ndarray, center: float) -> np.ndarray:
    """Gaussian bell on hue circle; half-width = _CB_HALF_WIDTH degrees."""
    diff = pixel_hue - center
    # Wrap to [-180, 180]
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.exp(-0.5 * (diff / _CB_HALF_WIDTH) ** 2)


def apply_color_blender(rgb: np.ndarray, bands) -> np.ndarray:
    """
    Per-hue HSL adjustment using 8 Gaussian-weighted bands.
    bands: list of 8 ColorBlenderBand(hue, chroma, brightness)
    """
    any_active = any(
        abs(b.hue) > 1e-6 or abs(b.chroma) > 1e-6 or abs(b.brightness) > 1e-6
        for b in bands
    )
    if not any_active:
        return rgb

    hsl = _rgb_to_hsl(rgb)
    h_px = hsl[:, 0]

    # Accumulate weighted deltas
    dh = np.zeros(len(rgb))
    dc = np.zeros(len(rgb))
    dl = np.zeros(len(rgb))
    w_total = np.zeros(len(rgb))

    for band, center in zip(bands, _CB_HUE_CENTERS):
        w = _hue_weight(h_px, center)
        dh += w * (band.hue / 100.0) * 60.0      # hue shift in degrees
        dc += w * (band.chroma / 100.0)
        dl += w * (band.brightness / 100.0) * 0.5
        w_total += w

    # Normalise so overlapping bands don't amplify beyond their individual maximum
    norm = np.where(w_total > 1.0, w_total, 1.0)
    dh /= norm
    dc /= norm
    dl /= norm

    hsl[:, 0] = (hsl[:, 0] + dh) % 360.0
    hsl[:, 1] = np.clip(hsl[:, 1] + dc, 0, 1)
    hsl[:, 2] = np.clip(hsl[:, 2] + dl, 0, 1)

    return _hsl_to_rgb(hsl)


def apply_color_grading(rgb: np.ndarray, grading) -> np.ndarray:
    """
    3-way color wheel (shadows / midtone / highlights).
    Each zone's hue and chroma define a color-cast vector in RGB.
    Luminance-based masks blend the three zones together.
    """
    cg = grading
    zones = [cg.shadows, cg.midtone, cg.highlights]

    any_active = any(
        abs(z.hue) > 1e-6 or abs(z.chroma) > 1e-6 or abs(z.brightness) > 1e-6
        for z in zones
    )
    if not any_active:
        return rgb

    luma = _luma(rgb)
    balance = cg.balance / 100.0   # -1..1: positive → push toward highlights
    blending = cg.blending / 100.0  # 0..1: sharpness of zone transitions

    # Zone masks (smooth, overlapping)
    shadow_mask = np.clip(1.0 - luma * 2.0 + balance, 0, 1) ** (1.0 + blending)
    highlight_mask = np.clip(luma * 2.0 - 1.0 - balance, 0, 1) ** (1.0 + blending)
    mid_mask = np.clip(1.0 - shadow_mask - highlight_mask, 0, 1)

    out = rgb.copy()
    for zone, mask in zip(zones, [shadow_mask, mid_mask, highlight_mask]):
        if abs(zone.chroma) < 1e-6 and abs(zone.brightness) < 1e-6:
            continue
        # Convert polar (hue°, chroma) to RGB bias
        hue_rad = math.radians(zone.hue)
        cr = math.cos(hue_rad) * zone.chroma / 100.0
        cg_b = math.sin(hue_rad) * zone.chroma / 100.0  # blue axis

        # Simple 3-way bias: R influenced by hue cosine, B by hue sine, G opposite
        r_bias = cr * 0.5
        g_bias = -abs(cr + cg_b) * 0.25
        b_bias = cg_b * 0.5

        # Brightness delta: ±0.15 max keeps midtone lifts photographic
        bright_delta = zone.brightness / 100.0 * 0.15

        bias = np.array([r_bias, g_bias, b_bias]) + bright_delta
        out += mask[:, np.newaxis] * bias

    return np.clip(out, 0, 1)
