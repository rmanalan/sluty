import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.fixtures.make_fixtures import (
    make_np3_neutral,
    make_np3_saturation50,
    make_ncp_neutral,
)
from npc_to_lut.parser import parse_np3, parse_ncp


def test_np3_neutral_scalars():
    pc = parse_np3(make_np3_neutral())
    assert pc.source_format == "np3"
    assert pc.name == "Neutral"
    assert pc.contrast == 0.0
    assert pc.saturation == 0.0
    assert pc.highlights == 0.0
    assert pc.shadows == 0.0
    assert pc.white_level == 0.0
    assert pc.black_level == 0.0


def test_np3_neutral_tone_curve_is_identity():
    """When tone curve flag=0 no custom curve is set."""
    pc = parse_np3(make_np3_neutral())
    assert pc.tone_curve is None


def test_np3_saturation_decoded():
    pc = parse_np3(make_np3_saturation50())
    assert pc.saturation == 50.0


def test_np3_color_blender_neutral():
    pc = parse_np3(make_np3_neutral())
    for band in pc.color_blender:
        assert band.hue == 0.0
        assert band.chroma == 0.0
        assert band.brightness == 0.0


def test_np3_color_grading_neutral():
    pc = parse_np3(make_np3_neutral())
    for zone in [pc.color_grading.highlights, pc.color_grading.midtone, pc.color_grading.shadows]:
        assert zone.hue == 0.0
        assert zone.chroma == 0.0
        assert zone.brightness == 0.0


def test_ncp_base_curve():
    pc = parse_ncp(make_ncp_neutral())
    assert pc.source_format == "ncp"
    assert pc.base_curve_id == 0x03C2   # Neutral


def test_np3_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        parse_np3(b"\x00" * 10)
