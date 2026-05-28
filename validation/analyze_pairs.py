#!/usr/bin/env python3
"""
analyze_pairs.py — Pre-reduce all flat/styled pairs ONCE and cache to disk.

Reads each 146 MB TIF a single time, accumulates per-image LUT-cell statistics
(counts, sum_src, sum_dst), keeps a random pixel sample + a downscaled copy for
later cross-validation / visual QA, and reports coverage and the irreducible
in-gamut floor (the best any 3D LUT could do) — split into smooth vs edge so we
can confirm the 'sharpening tax' shrank now that Sharpening/Clarity/ADL are off.

Outputs a cache to CACHE_DIR for step 2 (cross-validation) without re-reading TIFs.
"""

import sys
import numpy as np
from pathlib import Path

from sluty import core as T

LUT_SIZE = 33
CACHE_DIR = Path("/tmp/npc_lut_cache")
SAMPLE_N = 200_000      # pixels kept per image for CV
DS = 8                  # downscale factor for visual / edge analysis


def rgb_err(a, b):
    return np.linalg.norm(a - b, axis=1) * 255.0


def cell_index(rgb, n):
    r = np.clip(np.round(rgb[:, 0] * (n - 1)), 0, n - 1).astype(np.int64)
    g = np.clip(np.round(rgb[:, 1] * (n - 1)), 0, n - 1).astype(np.int64)
    b = np.clip(np.round(rgb[:, 2] * (n - 1)), 0, n - 1).astype(np.int64)
    return (b * n + g) * n + r


def main():
    flat_dir = Path(sys.argv[1])
    styled_dir = Path(sys.argv[2])
    pairs = T.pairs_from_dirs(str(flat_dir), str(styled_dir))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n_cells = LUT_SIZE ** 3
    comb_counts = np.zeros(n_cells, np.int64)
    comb_ssrc = np.zeros((n_cells, 3), np.float64)
    comb_sdst = np.zeros((n_cells, 3), np.float64)

    print(f"{'image':<26}{'cov%':>7}{'floor':>8}{'smooth':>8}{'edge':>7}")
    print("-" * 56)

    per_image = []
    for i, (fp, sp) in enumerate(pairs):
        name = Path(fp).name
        flat = T.read_tif_float32(fp)
        chrome = T.read_tif_float32(sp)
        H, W, _ = flat.shape
        fpx = flat.reshape(-1, 3)
        cpx = chrome.reshape(-1, 3)

        c, ss, sd = T.accumulate_cells(fpx, cpx, LUT_SIZE)
        comb_counts += c; comb_ssrc += ss; comb_sdst += sd
        cov = 100 * (c > 0).sum() / n_cells

        # Per-image irreducible floor = pixel vs its own-image cell mean
        idx = cell_index(fpx, LUT_SIZE)
        cmean = sd[idx] / np.maximum(c[idx, None], 1)
        floor = rgb_err(cmean, cpx)

        # edge vs smooth split on this image
        lum = flat.mean(2)
        grad = (np.abs(np.diff(lum, axis=1, prepend=lum[:, :1])) +
                np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))).ravel()
        smooth = grad < np.percentile(grad, 50)
        edge = grad > np.percentile(grad, 90)
        print(f"{name:<26}{cov:>7.1f}{floor.mean():>8.2f}"
              f"{floor[smooth].mean():>8.2f}{floor[edge].mean():>7.2f}")

        # cache a random pixel sample for CV
        sel = rng.choice(len(fpx), SAMPLE_N, replace=False)
        np.savez(CACHE_DIR / f"img_{i:02d}.npz",
                 name=name,
                 counts=c, ssrc=ss, sdst=sd,
                 samp_flat=fpx[sel].astype(np.float32),
                 samp_chrome=cpx[sel].astype(np.float32),
                 ds_flat=flat[::DS, ::DS].astype(np.float32),
                 ds_chrome=chrome[::DS, ::DS].astype(np.float32))
        per_image.append((name, cov, floor.mean()))
        del flat, chrome, fpx, cpx

    np.savez(CACHE_DIR / "combined.npz",
             counts=comb_counts, ssrc=comb_ssrc, sdst=comb_sdst,
             n_pairs=len(pairs))

    comb_cov = 100 * (comb_counts > 0).sum() / n_cells
    print("-" * 56)
    print(f"{'COMBINED (all '+str(len(pairs))+')':<26}{comb_cov:>7.1f}")
    print(f"\nSingle-image coverage range: "
          f"{min(p[1] for p in per_image):.1f}%–{max(p[1] for p in per_image):.1f}%")
    print(f"Combined coverage: {comb_cov:.1f}%  "
          f"(vs ~4% for the old single sharpened photo)")
    print(f"\nMean per-image floor: {np.mean([p[2] for p in per_image]):.2f}")
    print("  (compare to 14.15 on the OLD sharpening-ON data — lower = sharpening-off paid off)")
    print(f"\nCache written to {CACHE_DIR}")


if __name__ == "__main__":
    main()
