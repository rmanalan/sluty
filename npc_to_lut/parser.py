"""
Binary parser for Nikon Picture Control files.

Supported formats:
  .NP3  — Flexible Color Picture Control (Zf, Z8, Z9, Z6III …)
  .NCP  — Classic Custom Picture Control (D-series, older Z-series)
  .NPC  — Marketing alias; treated as NP3 when size >= 0x400, else NCP.

Format references:
  NP3 offsets:  github.com/ssssota/nikon-flexible-color-picture-control
  NCP offsets:  github.com/horshack-dpreview/NikonPictureControlsDev
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ColorBlenderBand:
    hue: float        # -100 .. 100
    chroma: float     # -100 .. 100
    brightness: float # -100 .. 100


@dataclass
class ColorGradingZone:
    hue: float        # 0 .. 360
    chroma: float     # -100 .. 100
    brightness: float # -100 .. 100


@dataclass
class ColorGrading:
    highlights: ColorGradingZone = field(default_factory=lambda: ColorGradingZone(0, 0, 0))
    midtone: ColorGradingZone    = field(default_factory=lambda: ColorGradingZone(0, 0, 0))
    shadows: ColorGradingZone    = field(default_factory=lambda: ColorGradingZone(0, 0, 0))
    blending: float = 50.0   # 0 .. 100
    balance: float  = 0.0    # -100 .. 100


@dataclass
class PictureControl:
    name: str = "Unnamed"
    source_format: str = "unknown"   # "np3" | "ncp"

    # Tonal adjustments
    contrast: float    = 0.0   # -100 .. 100
    highlights: float  = 0.0   # -100 .. 100
    shadows: float     = 0.0   # -100 .. 100
    white_level: float = 0.0   # -100 .. 100
    black_level: float = 0.0   # -100 .. 100

    # Chroma
    saturation: float = 0.0    # -100 .. 100

    # Tone curve: 257 floats in [0, 1] (identity = linear ramp).
    # Derived from the 257-entry uint16 LUT (0..32767) or from spline points.
    tone_curve: Optional[list] = None  # None → identity

    # NCP base curve enum (0x0001=Standard, 0x03C2=Neutral, 0x00C3=Vivid)
    base_curve_id: Optional[int] = None

    # Color Blender (8 hue bands): Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta
    color_blender: list = field(default_factory=lambda: [
        ColorBlenderBand(0, 0, 0) for _ in range(8)
    ])

    # Color Grading (3-way color wheel)
    color_grading: ColorGrading = field(default_factory=ColorGrading)


# ---------------------------------------------------------------------------
# NP3 parser
# ---------------------------------------------------------------------------

_NP3_OFF_NAME           = 0x18
_NP3_OFF_SHARPENING     = 0x52
_NP3_OFF_MID_SHARP      = 0xF2
_NP3_OFF_CLARITY        = 0x5C
_NP3_OFF_CONTRAST       = 0x110
_NP3_OFF_HIGHLIGHTS     = 0x11A
_NP3_OFF_SHADOWS        = 0x124
_NP3_OFF_WHITE_LEVEL    = 0x12E
_NP3_OFF_BLACK_LEVEL    = 0x138
_NP3_OFF_SATURATION     = 0x142
_NP3_OFF_CB_RED         = 0x14C   # color blender; 3 bytes per band
_NP3_OFF_CG_HIGHLIGHTS  = 0x170   # color grading; 4 bytes per zone
_NP3_OFF_CG_MIDTONE     = 0x174
_NP3_OFF_CG_SHADOWS     = 0x178
_NP3_OFF_CG_BLENDING    = 0x180
_NP3_OFF_CG_BALANCE     = 0x182
_NP3_OFF_COMMENT_FLAG_A = 0x185
_NP3_OFF_COMMENT_FLAG_B = 0x186
_NP3_OFF_TC_FLAG        = 0x187
_NP3_OFF_TC_POINTS      = 0x194
_NP3_OFF_TC_RAW         = 0x1CC

_CB_BAND_STRIDE = 3   # bytes per color-blender band
_CB_BAND_COUNT  = 8   # Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta

# Minimum file size for a valid NP3 (tc_flag=0, no tone-curve data needed)
_NP3_MIN_SIZE = _NP3_OFF_TC_FLAG + 1   # 0x188 = 392 bytes


def _bias(v: int) -> float:
    """Decode a biased uint8 (bias = 0x80) to a signed integer."""
    return float(v - 0x80)


def _read_cg_zone(data: bytes, offset: int) -> ColorGradingZone:
    """
    4-byte color grading zone:
      byte 0 [bits 7..4]: high nibble of 10-bit hue
      byte 1            : low 8 bits of hue   → hue = ((b0 & 0x0F) << 8) | b1
      byte 2            : chroma  (biased uint8)
      byte 3            : brightness (biased uint8)
    """
    b0, b1, b2, b3 = struct.unpack_from("4B", data, offset)
    hue = ((b0 & 0x0F) << 8) | b1
    return ColorGradingZone(
        hue=float(hue),
        chroma=_bias(b2),
        brightness=_bias(b3),
    )


def _decode_tc_raw(data: bytes) -> list:
    """Read 257 big-endian uint16 values starting at _NP3_OFF_TC_RAW and normalise to [0,1]."""
    values = []
    for i in range(257):
        (v,) = struct.unpack_from(">H", data, _NP3_OFF_TC_RAW + i * 2)
        values.append(v / 32767.0)
    return values


def _decode_tc_points(data: bytes) -> list:
    """
    Spline points section at _NP3_OFF_TC_POINTS:
      byte 0: number of points (max 20)
      followed by pairs of (x, y) uint8 values
    Returns normalised [0,1] pairs.
    """
    n = struct.unpack_from("B", data, _NP3_OFF_TC_POINTS)[0]
    n = min(n, 20)
    pts = []
    for i in range(n):
        x, y = struct.unpack_from("2B", data, _NP3_OFF_TC_POINTS + 1 + i * 2)
        pts.append((x / 255.0, y / 255.0))
    return pts


def _spline_to_lut(points: list) -> list:
    """Convert up to 20 (x, y) spline points to a 257-entry LUT via scipy cubic spline."""
    import numpy as np
    from scipy.interpolate import CubicSpline

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Ensure endpoints are present
    if xs[0] > 0:
        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, 0.0)
    if xs[-1] < 1:
        xs = np.append(xs, 1.0)
        ys = np.append(ys, 1.0)

    cs = CubicSpline(xs, ys, bc_type="natural")
    t = np.linspace(0, 1, 257)
    return np.clip(cs(t), 0, 1).tolist()


def parse_np3(data: bytes) -> PictureControl:
    if len(data) < _NP3_MIN_SIZE:
        raise ValueError(f"NP3 file too short: {len(data)} bytes")

    pc = PictureControl(source_format="np3")

    # Name (null-terminated, up to 19 bytes)
    raw_name = data[_NP3_OFF_NAME: _NP3_OFF_NAME + 19]
    pc.name = raw_name.split(b"\x00")[0].decode("ascii", errors="replace")

    pc.contrast    = _bias(data[_NP3_OFF_CONTRAST])
    pc.highlights  = _bias(data[_NP3_OFF_HIGHLIGHTS])
    pc.shadows     = _bias(data[_NP3_OFF_SHADOWS])
    pc.white_level = _bias(data[_NP3_OFF_WHITE_LEVEL])
    pc.black_level = _bias(data[_NP3_OFF_BLACK_LEVEL])
    pc.saturation  = _bias(data[_NP3_OFF_SATURATION])

    # Tone curve
    tc_flag = data[_NP3_OFF_TC_FLAG]
    if tc_flag == 2:
        if len(data) < _NP3_OFF_TC_RAW + 257 * 2:
            raise ValueError(f"NP3 file too short for raw tone curve: {len(data)} bytes")
        pc.tone_curve = _decode_tc_raw(data)
    elif tc_flag == 1:
        pts = _decode_tc_points(data)
        if pts:
            pc.tone_curve = _spline_to_lut(pts)

    # Color Blender (8 bands × 3 bytes)
    bands = []
    for i in range(_CB_BAND_COUNT):
        off = _NP3_OFF_CB_RED + i * _CB_BAND_STRIDE
        h, c, b = struct.unpack_from("3B", data, off)
        bands.append(ColorBlenderBand(
            hue=_bias(h),
            chroma=_bias(c),
            brightness=_bias(b),
        ))
    pc.color_blender = bands

    # Color Grading
    cg = ColorGrading()
    cg.highlights = _read_cg_zone(data, _NP3_OFF_CG_HIGHLIGHTS)
    cg.midtone    = _read_cg_zone(data, _NP3_OFF_CG_MIDTONE)
    cg.shadows    = _read_cg_zone(data, _NP3_OFF_CG_SHADOWS)

    cg.blending  = _bias(data[_NP3_OFF_CG_BLENDING])
    cg.balance   = _bias(data[_NP3_OFF_CG_BALANCE])
    pc.color_grading = cg

    return pc


# ---------------------------------------------------------------------------
# NCP parser (classic format)
# ---------------------------------------------------------------------------

_NCP_OFF_BASE_CURVE   = 0x24
_NCP_MIN_SIZE_FOR_UDC = 0x80   # files with a UDC are longer than this

_NCP_BASE_CURVE_NAMES = {
    0x0001: "Standard",
    0x00C3: "Vivid",
    0x03C2: "Neutral",
}


def _find_ncp_udc(data: bytes) -> Optional[list]:
    """
    Locate the 257-entry UDC in an NCP file.  The format is not fully documented, but
    the UDC appears as 257 consecutive big-endian uint16 values with the property that
    udc[0] == 0 and udc[256] == 32767.  We scan from offset 0x40 for this signature.
    """
    for off in range(0x40, len(data) - 257 * 2 + 1, 2):
        v_first, = struct.unpack_from(">H", data, off)
        v_last,  = struct.unpack_from(">H", data, off + 256 * 2)
        if v_first == 0 and v_last == 32767:
            vals = []
            ok = True
            prev = -1
            for i in range(257):
                (v,) = struct.unpack_from(">H", data, off + i * 2)
                if v < prev:    # must be non-decreasing
                    ok = False
                    break
                vals.append(v / 32767.0)
                prev = v
            if ok:
                return vals
    return None


def parse_ncp(data: bytes) -> PictureControl:
    pc = PictureControl(source_format="ncp")

    if len(data) > _NCP_OFF_BASE_CURVE + 2:
        (bc_id,) = struct.unpack_from(">H", data, _NCP_OFF_BASE_CURVE)
        pc.base_curve_id = bc_id

    pc.tone_curve = _find_ncp_udc(data)

    return pc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load(path: str) -> PictureControl:
    """
    Load and parse a Nikon Picture Control file.
    Format is auto-detected by file size (NP3 >= _NP3_MIN_SIZE bytes, else NCP).
    Real NP3 files with tc_flag=0 (no custom tone curve) are ~640 bytes, well
    below the old 0x400 threshold that caused them to be mis-parsed as NCP.
    """
    data = Path(path).read_bytes()
    if len(data) >= _NP3_MIN_SIZE:
        return parse_np3(data)
    return parse_ncp(data)
