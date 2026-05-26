"""
Build a 3D LUT from a parsed PictureControl.

The LUT is stored as a numpy array of shape (size, size, size, 3)
where index order is (R, G, B) and values are in [0, 1].
In the .cube spec, R is the fastest-varying axis.
"""

import numpy as np
from .parser import PictureControl
from .transforms import (
    apply_tone_curve,
    apply_contrast,
    apply_highlights,
    apply_shadows,
    apply_white_black,
    apply_saturation,
    apply_color_blender,
    apply_color_grading,
)


def build_lut(pc: PictureControl, size: int = 33, debug: bool = False) -> np.ndarray:
    """
    Build a (size, size, size, 3) LUT by applying all color transforms
    from *pc* to an identity grid.

    If *debug* is True, prints per-stage max delta vs. identity.
    """
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)

    # .cube spec: R changes fastest, then G, then B
    # meshgrid 'ij' → shape (size_R, size_G, size_B)
    R, G, B = np.meshgrid(axis, axis, axis, indexing='ij')
    grid = np.stack([R, G, B], axis=-1)          # (size, size, size, 3)
    flat = grid.reshape(-1, 3)                    # (N, 3)

    def _stage(name: str, data: np.ndarray) -> np.ndarray:
        if debug:
            delta = np.abs(data - flat).max()
            print(f"  [{name:30s}] max delta from identity = {delta:.6f}")
        return data

    flat = _stage("identity",        flat.copy())
    flat = _stage("tone_curve",      apply_tone_curve(flat, pc.tone_curve))
    flat = _stage("contrast",        apply_contrast(flat, pc.contrast))
    flat = _stage("highlights",      apply_highlights(flat, pc.highlights))
    flat = _stage("shadows",         apply_shadows(flat, pc.shadows))
    flat = _stage("white/black",     apply_white_black(flat, pc.white_level, pc.black_level))
    flat = _stage("saturation",      apply_saturation(flat, pc.saturation))
    flat = _stage("color_blender",   apply_color_blender(flat, pc.color_blender))
    flat = _stage("color_grading",   apply_color_grading(flat, pc.color_grading))

    return np.clip(flat.reshape(size, size, size, 3), 0.0, 1.0)
