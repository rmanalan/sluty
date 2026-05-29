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


def _reversals(lut, n):
    L = lut.reshape(n, n, n, 3)
    return sum(int((np.diff(L, axis=ax)[..., ch] < -1e-4).sum())
               for ax, ch in ((2, 0), (1, 1), (0, 2)))


def test_enforce_monotonic_removes_reversals():
    # A steep, partly-inverted tone curve provokes extrapolation ringing.
    rng = np.random.default_rng(3)
    src = rng.random((4000, 3)).astype(np.float32)
    dst = np.clip(src + 0.3 * np.sin(src * 9.0), 0, 1).astype(np.float32)
    counts, ss, sd = core.accumulate_cells(src, dst, 17)

    base = core.fit_lut(counts, ss, sd, 17, verbose=False)
    guarded = core.fit_lut(counts, ss, sd, 17, verbose=False, enforce_monotonic=True)

    assert _reversals(base, 17) > 0          # baseline rings
    assert _reversals(guarded, 17) == 0      # guard removes every reversal
    assert guarded.shape == base.shape
    assert guarded.min() >= 0.0 and guarded.max() <= 1.0

    # Idempotent on already-monotonic input.
    again = core.enforce_monotonic_lut(guarded, 17)
    np.testing.assert_array_equal(again, guarded)


def test_enforce_monotonic_preserves_identity():
    # Guard must not distort a clean monotonic (identity) LUT.
    ident = core.cube_grid(9).astype(np.float32)
    out = core.enforce_monotonic_lut(ident, 9)
    np.testing.assert_allclose(out, ident, atol=1e-6)


def _chroma(lut):
    return np.sqrt(((lut - lut.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))


def test_shadow_desat_neutralizes_darks_only():
    # Hand-built nodes spanning luma 0 → bright, each carrying chroma. Desat should
    # scale chroma by exactly (1 - weight): full kill at luma 0, none above `hi`.
    hi = 0.18
    # Pure-blue nodes at increasing brightness (luma = 0.0722 * blue level).
    lut = np.array([[0.0, 0.0, b] for b in (0.0, 0.5, 1.0)] +
                   [[0.6, 0.6, 1.0]],                      # bright, luma > hi
                   dtype=np.float32)
    luma = lut @ np.array([0.2126, 0.7152, 0.0722])
    w = np.clip((hi - luma) / hi, 0.0, 1.0)

    out = core.shadow_desaturate_lut(lut, hi)

    # Black node (luma 0, w = 1) is fully neutral; bright node (luma > hi) untouched.
    assert _chroma(out)[0] < 1e-6
    np.testing.assert_allclose(out[-1], lut[-1], atol=1e-6)
    # Every node's chroma scaled by exactly (1 - w).
    np.testing.assert_allclose(_chroma(out), _chroma(lut) * (1.0 - w), atol=1e-5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_shadow_desat_zero_is_noop():
    grid = core.cube_grid(9)
    lut = np.clip(grid + 0.1, 0, 1).astype(np.float32)
    np.testing.assert_array_equal(core.shadow_desaturate_lut(lut, 0.0), lut)
