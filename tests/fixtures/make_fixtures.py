"""
Generate synthetic .NP3 and .NCP fixture files for testing.
Run once: python tests/fixtures/make_fixtures.py
"""

import struct
from pathlib import Path

HERE = Path(__file__).parent


def make_np3_neutral() -> bytes:
    """
    NP3 fixture with ALL adjustments at zero / identity.
    Must be >= 0x400 bytes.
    """
    data = bytearray(0x500)

    # Name "Neutral"
    name = b"Neutral\x00"
    data[0x18: 0x18 + len(name)] = name

    # All scalar adjustments at 0x80 (= zero after bias subtraction)
    for off in [0x52, 0xF2, 0x5C, 0x110, 0x11A, 0x124, 0x12E, 0x138, 0x142]:
        data[off] = 0x80

    # Color blender: 8 bands × 3 bytes, all 0x80
    for i in range(8 * 3):
        data[0x14C + i] = 0x80

    # Color grading: 3 zones × 4 bytes; hue=0 (2 bytes), chroma=0x80, brightness=0x80
    for zone_off in [0x170, 0x174, 0x178]:
        data[zone_off]     = 0x00
        data[zone_off + 1] = 0x00
        data[zone_off + 2] = 0x80
        data[zone_off + 3] = 0x80

    # CG blending / balance
    struct.pack_into(">H", data, 0x180, 0x80)
    data[0x182] = 0x80

    # Tone curve: flag=0 (no custom curve)
    data[0x187] = 0x00

    # Identity tone curve raw (257 × uint16 big-endian, linearly 0..32767)
    for i in range(257):
        val = round(i * 32767 / 256)
        struct.pack_into(">H", data, 0x1CC + i * 2, val)

    return bytes(data)


def make_np3_saturation50() -> bytes:
    """NP3 fixture with saturation = +50, all else neutral."""
    data = bytearray(make_np3_neutral())
    data[0x18: 0x18 + 8] = b"Sat50\x00\x00\x00"
    data[0x142] = 0x80 + 50    # saturation = +50
    return bytes(data)


def make_np3_contrast30() -> bytes:
    """NP3 fixture with contrast = +30, all else neutral."""
    data = bytearray(make_np3_neutral())
    data[0x18: 0x18 + 8] = b"Ctr30\x00\x00\x00"
    data[0x110] = 0x80 + 30
    return bytes(data)


def make_np3_s_curve() -> bytes:
    """NP3 fixture with a custom S-curve tone curve (raw mode)."""
    import math
    data = bytearray(make_np3_neutral())
    data[0x18: 0x18 + 8] = b"SCurve\x00\x00"
    data[0x187] = 2   # raw tone curve

    # S-curve: f(x) = 0.5 + tanh(3*(x-0.5)) / (2*tanh(1.5))
    k = 2.0 * math.tanh(1.5)
    for i in range(257):
        x = i / 256.0
        y = 0.5 + math.tanh(3.0 * (x - 0.5)) / k
        val = round(max(0, min(1, y)) * 32767)
        struct.pack_into(">H", data, 0x1CC + i * 2, val)

    return bytes(data)


def make_ncp_neutral() -> bytes:
    """Minimal NCP fixture (< 0x400 bytes) with Neutral base curve and no UDC."""
    data = bytearray(0x80)
    struct.pack_into(">H", data, 0x24, 0x03C2)  # Neutral
    return bytes(data)


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "neutral.NP3").write_bytes(make_np3_neutral())
    (HERE / "saturation50.NP3").write_bytes(make_np3_saturation50())
    (HERE / "contrast30.NP3").write_bytes(make_np3_contrast30())
    (HERE / "scurve.NP3").write_bytes(make_np3_s_curve())
    (HERE / "neutral.NCP").write_bytes(make_ncp_neutral())
    print("Fixtures written to", HERE)
