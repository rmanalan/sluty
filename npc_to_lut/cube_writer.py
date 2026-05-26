"""
Write a numpy LUT array to the Adobe .cube format.

Spec reference: Adobe Cube LUT Specification 1.0
  - LUT_3D_SIZE n  → n×n×n entries
  - Fastest-varying axis: R, then G, then B
  - Each line: R G B as space-separated floats
"""

import numpy as np
from pathlib import Path


def write_cube(lut: np.ndarray, path: str, title: str = "NPC LUT") -> None:
    """
    Write *lut* (shape: size×size×size×3, values in [0,1]) to a .cube file at *path*.
    """
    size = lut.shape[0]
    if lut.shape != (size, size, size, 3):
        raise ValueError(f"LUT must be (S,S,S,3); got {lut.shape}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]

    # .cube: iterate B (outer) → G → R (inner)
    # Our array indexing: lut[r_idx, g_idx, b_idx]
    for b_i in range(size):
        for g_i in range(size):
            for r_i in range(size):
                r, g, b = lut[r_i, g_i, b_i]
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")

    out.write_text("\n".join(lines) + "\n", encoding="ascii")
