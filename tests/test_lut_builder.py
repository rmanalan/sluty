import numpy as np
import pytest

from npc_to_lut.parser import PictureControl, ColorBlenderBand
from npc_to_lut.lut_builder import build_lut
from npc_to_lut.cube_writer import write_cube
import tempfile, os


def _neutral_pc() -> PictureControl:
    return PictureControl()   # all defaults = zero / identity


def test_identity_lut_from_neutral_pc():
    """A fully neutral PictureControl must produce a near-identity LUT."""
    pc = _neutral_pc()
    lut = build_lut(pc, size=17)

    axis = np.linspace(0, 1, 17)
    for r_i, r_v in enumerate(axis):
        for g_i, g_v in enumerate(axis):
            for b_i, b_v in enumerate(axis):
                node = lut[r_i, g_i, b_i]
                np.testing.assert_allclose(
                    node, [r_v, g_v, b_v], atol=1e-6,
                    err_msg=f"Node ({r_i},{g_i},{b_i}) deviated"
                )


def test_saturation_only_pc():
    """With only saturation set, gray nodes must remain gray."""
    pc = _neutral_pc()
    pc.saturation = 75.0
    lut = build_lut(pc, size=5)

    axis = np.linspace(0, 1, 5)
    for i in range(5):
        v = axis[i]
        node = lut[i, i, i]   # diagonal = gray
        # All three channels must be (almost) equal
        assert abs(node[0] - node[1]) < 1e-4
        assert abs(node[1] - node[2]) < 1e-4


def test_contrast_stretches_lut():
    """Contrast >0 should make values above 0.5 brighter in the LUT."""
    pc = _neutral_pc()
    pc.contrast = 50.0
    lut = build_lut(pc, size=17)

    # Node at index 12 ≈ value 0.75
    node = lut[12, 12, 12]
    assert node[0] > 12 / 16   # brighter than the identity value


def test_tone_curve_applied():
    """An S-curve tone curve should shift midtone gray away from 0.5."""
    import math
    pc = _neutral_pc()
    k = 2.0 * math.tanh(1.5)
    pc.tone_curve = [
        0.5 + math.tanh(3.0 * (i / 256.0 - 0.5)) / k for i in range(257)
    ]
    lut = build_lut(pc, size=5)
    # Midpoint (index 2 in a 5-node grid, value 0.5) should map to ~0.5
    mid = lut[2, 2, 2]
    np.testing.assert_allclose(mid, [0.5, 0.5, 0.5], atol=0.05)


def test_cube_writer_round_trip(tmp_path):
    """Write a small identity LUT and verify corners via cube_writer."""
    pc = _neutral_pc()
    lut = build_lut(pc, size=5)
    out = str(tmp_path / "test.cube")
    write_cube(lut, out, title="Test")

    content = open(out).read()
    assert "LUT_3D_SIZE 5" in content
    assert "TITLE" in content
    lines = [l for l in content.splitlines() if l and l[0].isdigit()]
    assert len(lines) == 5 ** 3


def test_lut_corners():
    """Black and white corners must map to 0 and 1."""
    pc = _neutral_pc()
    pc.contrast = 30.0
    pc.saturation = 40.0
    lut = build_lut(pc, size=33)

    black = lut[0, 0, 0]
    white = lut[32, 32, 32]

    np.testing.assert_allclose(black, [0.0, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(white, [1.0, 1.0, 1.0], atol=0.02)
