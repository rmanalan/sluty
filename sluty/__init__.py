"""sLUTy — derive a 3D .cube LUT from before/after image pairs."""

from .core import (
    read_image_float32,
    cube_grid,
    accumulate_cells,
    fit_lut,
    apply_lut,
    validate_gamut_corners,
    write_cube,
    pairs_from_dirs,
    derive_lut,
)

__version__ = "0.1.0"

__all__ = [
    "read_image_float32",
    "cube_grid",
    "accumulate_cells",
    "fit_lut",
    "apply_lut",
    "validate_gamut_corners",
    "write_cube",
    "pairs_from_dirs",
    "derive_lut",
]
