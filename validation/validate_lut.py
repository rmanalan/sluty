#!/usr/bin/env python3
"""
validate_lut.py — Prove which LUT-derivation method best reconstructs a
Picture-Control-like transform from limited photographic coverage.

Why synthetic: with the real Nikon data we only ever know the answer for the
~4% of colours that appear in the photo. Here we DEFINE a known transform T,
so we know the correct output for EVERY colour — including the saturated
gamut corners no photograph fills. That lets us measure the one thing that
actually matters: error in the unmeasured regions.

Pipeline:
  1. Define a realistic film-look transform T (tone curve + saturation +
     hue twist + channel cross-talk + black lift).
  2. Synthesize a "flat" image whose colours mimic a real photograph's
     limited gamut (a handful of Gaussian blobs: skin, foliage, sky, neutrals).
  3. styled = T(flat), quantized to mimic export noise.
  4. Derive a LUT from that single pair via each method.
  5. Apply each LUT to a DENSE uniform sweep of the full cube and compare to
     the true T. Report error overall and split by in-gamut vs out-of-gamut
     (near training colours vs far from them).
"""

import numpy as np
from scipy.spatial import cKDTree

from sluty import core as T


# ---------------------------------------------------------------------------
# Ground-truth "Picture Control" transform
# ---------------------------------------------------------------------------

def _tone_curve(x):
    """Filmic-ish S-curve with a lifted black toe (per-channel, on [0,1])."""
    # lifted toe + gentle shoulder
    y = 0.5 + 1.35 * (x - 0.5)            # contrast
    y = np.clip(y, 0.0, 1.0)
    y = 0.04 + (1.0 - 0.04 - 0.02) * y    # black lift 0.04, white ceiling 0.98
    return y


def _rgb_to_hsv(rgb):
    """Vectorized RGB→HSV, all in [0,1] (H normalized to [0,1))."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(1); mn = rgb.min(1); d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-12
    rm = nz & (mx == r); gm = nz & (mx == g) & ~rm; bm = nz & (mx == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4
    h /= 6.0
    s = np.where(mx > 1e-12, d / np.maximum(mx, 1e-12), 0.0)
    return np.stack([h, s, mx], axis=1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0] * 6.0, hsv[:, 1], hsv[:, 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    out = np.zeros_like(hsv)
    for k, (R, G, B) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m = i == k
        out[m, 0] = R[m]; out[m, 1] = G[m]; out[m, 2] = B[m]
    return out


def _piecewise_tone(x):
    """Piecewise-linear tone curve with kinks — deliberately NON-smooth (LUT-like)."""
    xp = np.array([0.0, 0.18, 0.45, 0.72, 1.0])
    fp = np.array([0.05, 0.12, 0.46, 0.82, 0.97])
    return np.interp(x, xp, fp)


def true_transform_hueband(rgb):
    """
    A deliberately LUT-like 'film' transform that is HARDER for a smooth RBF:
    piecewise-linear tone curve + hue-banded saturation/shift with sharpish
    boundaries (the kind of move Nikon's Color Blender makes). Tests whether
    RBF's advantage survives a non-polynomial transform.
    """
    rgb = np.clip(rgb, 0, 1)
    # piecewise tone per channel
    out = np.stack([_piecewise_tone(rgb[:, c]) for c in range(3)], axis=1)
    hsv = _rgb_to_hsv(out)
    h = hsv[:, 0]
    # hue-banded saturation: boost oranges (~0.05-0.12), cut greens (~0.25-0.45)
    sat = hsv[:, 1].copy()
    orange = (h > 0.03) & (h < 0.13)
    green  = (h > 0.22) & (h < 0.45)
    sat[orange] *= 1.35
    sat[green]  *= 0.70
    hsv[:, 1] = np.clip(sat, 0, 1)
    # small hue push: rotate blues toward teal
    blue = (h > 0.55) & (h < 0.75)
    hsv[blue, 0] = (hsv[blue, 0] - 0.03) % 1.0
    out = _hsv_to_rgb(hsv)
    # global black lift + white ceiling
    return np.clip(0.02 + out * 0.96, 0, 1)


def true_transform(rgb):
    """
    A plausible 'Documentary Chrome'-style look. Input/output sRGB-ish [0,1].
    Deterministic, smooth, nonlinear in colour — the kind of thing a 3D LUT
    is meant to capture.
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # 1) per-channel tone curve with slight channel-specific ceilings (teal/orange split)
    r2 = _tone_curve(r) * 1.00
    g2 = _tone_curve(g) * 0.99
    b2 = _tone_curve(b) * 0.97

    out = np.stack([r2, g2, b2], axis=1)

    # 2) luminance-preserving saturation drop (film desat ~ -15%)
    luma = (out * np.array([0.2126, 0.7152, 0.0722])).sum(1, keepdims=True)
    out = luma + (out - luma) * 0.85

    # 3) hue twist via a fixed 3x3 channel mixer (teal shadows / warm mids)
    M = np.array([
        [1.02, 0.02, -0.04],
        [-0.03, 1.01, 0.02],
        [0.02, -0.05, 1.03],
    ])
    out = out @ M.T

    # 4) split-tone: warm highlights, cool shadows (luminance-masked)
    l = (out * np.array([0.2126, 0.7152, 0.0722])).sum(1, keepdims=True)
    warm = np.array([0.03, 0.01, -0.03])   # highlights push warm
    cool = np.array([-0.02, 0.0, 0.04])    # shadows push cool
    hi_mask = np.clip(l, 0, 1)
    out = out + hi_mask * warm + (1 - hi_mask) * cool

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Synthetic "photograph" with realistic limited gamut
# ---------------------------------------------------------------------------

def synth_flat_image(n_pixels=2_000_000, seed=7, spread=1.0):
    """
    Sample colours from a few Gaussian blobs in RGB to mimic a real photo's
    gamut: neutrals, skin, foliage, sky, plus a small colourful minority.
    `spread` scales blob widths to dial coverage (smaller = more realistic,
    tighter gamut). Returns (n_pixels, 3) in [0,1].
    """
    rng = np.random.default_rng(seed)
    blobs = [
        # (center RGB,         covariance scale, weight)
        (np.array([0.55, 0.52, 0.50]), 0.10, 0.30),  # neutrals / midtones
        (np.array([0.70, 0.52, 0.42]), 0.06, 0.20),  # skin
        (np.array([0.30, 0.42, 0.22]), 0.07, 0.18),  # foliage
        (np.array([0.45, 0.58, 0.78]), 0.07, 0.15),  # sky
        (np.array([0.20, 0.20, 0.22]), 0.06, 0.10),  # shadows
        (np.array([0.85, 0.80, 0.70]), 0.05, 0.05),  # highlights
        (np.array([0.65, 0.30, 0.25]), 0.05, 0.02),  # the rare saturated stuff
    ]
    blobs = [(c, s * spread, w) for c, s, w in blobs]
    weights = np.array([w for *_, w in blobs])
    weights /= weights.sum()
    counts = rng.multinomial(n_pixels, weights)

    chunks = []
    for (center, scale, _), c in zip(blobs, counts):
        pts = rng.normal(center, scale, size=(c, 3))
        chunks.append(pts)
    flat = np.clip(np.concatenate(chunks, axis=0), 0.0, 1.0)
    rng.shuffle(flat)
    return flat


def quantize(rgb, bits=16, noise=0.0, seed=1):
    """Mimic export quantization (+ optional demosaic-ish noise)."""
    rng = np.random.default_rng(seed)
    if noise > 0:
        rgb = rgb + rng.normal(0, noise, rgb.shape)
    levels = (1 << bits) - 1
    return np.clip(np.round(np.clip(rgb, 0, 1) * levels) / levels, 0, 1)


# ---------------------------------------------------------------------------
# Apply a flat (n_cells,3) cube-order LUT via trilinear interpolation
# ---------------------------------------------------------------------------

def apply_lut(lut_flat, lut_size, rgb):
    """Trilinear interpolation through a .cube-order LUT (R fast, B slow)."""
    from scipy.interpolate import RegularGridInterpolator
    t = np.linspace(0, 1, lut_size)
    # cube order flat = b*N² + g*N + r  →  reshape to (B, G, R, 3); index by (b,g,r)
    vol = lut_flat.reshape(lut_size, lut_size, lut_size, 3)
    interp = RegularGridInterpolator((t, t, t), vol, bounds_error=False, fill_value=None)
    pts = np.stack([rgb[:, 2], rgb[:, 1], rgb[:, 0]], axis=1)  # (b, g, r)
    return np.clip(interp(pts), 0, 1)


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------

def rgb_err(a, b):
    """Mean per-pixel Euclidean RGB error, scaled to 0..255 for readability."""
    return np.linalg.norm(a - b, axis=1) * 255.0


def run_at_coverage(spread, label, methods, transform=None, lut_size=33):
    transform = transform or true_transform
    rng = np.random.default_rng(0)
    flat = synth_flat_image(n_pixels=2_000_000, spread=spread)
    # 16-bit quantization only (matches a real 16-bit TIF export); no extra
    # noise, so the comparison against clean truth is fair to both methods.
    styled = quantize(transform(flat), bits=16, noise=0.0)

    counts, sum_src, sum_dst = T.accumulate_cells(flat, styled, lut_size)
    pop_frac = 100 * (counts > 0).sum() / lut_size**3

    n_side = 40
    g = np.linspace(0, 1, n_side)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    test = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1)
    # measure against the same 16-bit quantized target the LUT was trained on
    truth = quantize(transform(test), bits=16, noise=0.0)

    tree = cKDTree(flat[rng.choice(len(flat), 50_000, replace=False)])
    dist, _ = tree.query(test, k=1)
    in_gamut = dist < 0.05
    out_gamut = ~in_gamut

    print(f"\n=== {label}: {pop_frac:.1f}% cells populated, "
          f"{100*out_gamut.mean():.0f}% of cube unmeasured ===")
    print(f"{'method':<38} {'overall':>9} {'in-gamut':>9} {'OUT-gamut':>10} {'max':>7}")
    print("-" * 78)
    results = []
    for mlabel, kw in methods:
        lut = T.fit_lut(counts, sum_src, sum_dst, lut_size, verbose=False, **kw)
        pred = apply_lut(lut, lut_size, test)
        e = rgb_err(pred, truth)
        row = (e.mean(), e[in_gamut].mean(), e[out_gamut].mean(), e.max())
        results.append((mlabel, row))
        print(f"{mlabel:<38} {row[0]:>9.2f} {row[1]:>9.2f} {row[2]:>10.2f} {row[3]:>7.2f}")

    nn = next(r for l, r in results if l.startswith("nn"))
    best_label, best = min(((l, r) for l, r in results if l.startswith("rbf")),
                           key=lambda x: x[1][0])
    print(f"  → best RBF ({best_label.strip()}): overall {nn[0]:.2f}→{best[0]:.2f} "
          f"({100*(1-best[0]/nn[0]):+.0f}%), "
          f"OUT-gamut {nn[2]:.2f}→{best[2]:.2f} ({100*(1-best[2]/nn[2]):+.0f}%)")
    return results


def main():
    methods = [
        ("nn  (old: nearest-neighbour fill)", dict(method="nn")),
        ("rbf thin_plate_spline",             dict(method="rbf", kernel="thin_plate_spline")),
        ("rbf linear",                        dict(method="rbf", kernel="linear")),
        ("rbf multiquadric",                  dict(method="rbf", kernel="multiquadric")),
        ("rbf cubic",                         dict(method="rbf", kernel="cubic")),
    ]
    print("Synthetic ground-truth validation (mean RGB error, 0–255 units; lower = better)")
    print("\n########## TRANSFORM A: smooth (tone + matrix + split-tone) ##########")
    run_at_coverage(0.45, "Tight gamut  (~one modest photo)", methods, true_transform)
    run_at_coverage(1.0,  "Broad gamut  (~several colourful photos)", methods, true_transform)

    print("\n########## TRANSFORM B: LUT-like (piecewise tone + hue-banded sat) ##########")
    print("# (adversarial: non-smooth transform that should be HARDER for RBF)")
    run_at_coverage(0.45, "Tight gamut  (~one modest photo)", methods, true_transform_hueband)
    run_at_coverage(1.0,  "Broad gamut  (~several colourful photos)", methods, true_transform_hueband)


if __name__ == "__main__":
    main()
