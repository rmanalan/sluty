import numpy as np
import pytest

from sluty import core


def _rand_image(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)).astype(np.float32)


def test_cube_grid_ordering():
    """R varies fastest, B slowest (Adobe .cube convention)."""
    g = core.cube_grid(3)
    assert g.shape == (27, 3)
    # first three nodes step in R only
    assert np.allclose(g[0], [0.0, 0.0, 0.0])
    assert np.allclose(g[1], [0.5, 0.0, 0.0])
    assert np.allclose(g[2], [1.0, 0.0, 0.0])
    # fourth node steps G
    assert np.allclose(g[3], [0.0, 0.5, 0.0])
    # tenth node (index 9) steps B
    assert np.allclose(g[9], [0.0, 0.0, 0.5])


def test_identity_pair_gives_identity_lut():
    """source == target ⇒ applying the LUT is (near) identity."""
    src = _rand_image()
    counts, ss, sd = core.accumulate_cells(src, src.copy(), 17)
    lut = core.fit_lut(counts, ss, sd, 17, verbose=False)
    probe = _rand_image(5000, seed=1)
    out = core.apply_lut(lut, 17, probe)
    assert np.abs(out - probe).mean() < 0.02


def test_known_transform_is_recovered_in_gamut():
    """A smooth per-channel transform is reproduced where data exists."""
    src = _rand_image(50000)
    # target: gentle per-channel curve + channel mix
    tgt = np.clip(np.stack([
        src[:, 0] ** 0.8,
        0.5 + (src[:, 1] - 0.5) * 1.2,
        src[:, 2] * 0.9 + 0.05,
    ], axis=1), 0, 1).astype(np.float32)
    counts, ss, sd = core.accumulate_cells(src, tgt, 33)
    lut = core.fit_lut(counts, ss, sd, 33, verbose=False)
    out = core.apply_lut(lut, 33, src)
    err = np.linalg.norm(out - tgt, axis=1).mean() * 255
    assert err < 5.0, f"in-gamut reconstruction error too high: {err:.2f}"


def test_nn_and_rbf_both_run():
    src = _rand_image()
    tgt = np.clip(src * 0.8 + 0.1, 0, 1).astype(np.float32)
    counts, ss, sd = core.accumulate_cells(src, tgt, 17)
    for method in ("rbf", "nn"):
        lut = core.fit_lut(counts, ss, sd, 17, method=method, verbose=False)
        assert lut.shape == (17 ** 3, 3)
        assert lut.min() >= 0.0 and lut.max() <= 1.0


def test_gamut_corner_check_clean_for_identity():
    src = _rand_image()
    counts, ss, sd = core.accumulate_cells(src, src.copy(), 17)
    lut = core.fit_lut(counts, ss, sd, 17, verbose=False)
    assert core.validate_gamut_corners(lut, 17, verbose=False) == []


def test_write_cube_roundtrip(tmp_path):
    src = _rand_image()
    counts, ss, sd = core.accumulate_cells(src, src.copy(), 9)
    lut = core.fit_lut(counts, ss, sd, 9, verbose=False)
    out = tmp_path / "x.cube"
    core.write_cube(lut, str(out), 9, "test")
    text = out.read_text().splitlines()
    assert any(l.startswith("LUT_3D_SIZE 9") for l in text)
    data = np.array([list(map(float, l.split())) for l in text
                     if l[:1].isdigit() or l[:1] == "-"])
    assert data.shape == (9 ** 3, 3)


def test_empty_input_raises():
    counts = np.zeros(17 ** 3, dtype=np.int64)
    ss = np.zeros((17 ** 3, 3))
    sd = np.zeros((17 ** 3, 3))
    with pytest.raises(ValueError, match="No populated cells"):
        core.fit_lut(counts, ss, sd, 17, verbose=False)
