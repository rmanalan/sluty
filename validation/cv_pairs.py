#!/usr/bin/env python3
"""
cv_pairs.py — Leave-one-image-out cross-validation on cached pair statistics.

For each held-out image: fit the LUT on the OTHER images only, then apply it to
the held-out image's pixels and compare to the real Documentary Chrome output.
This is the honest measure of how faithfully the LUT reproduces the picture
control on a photo it has NEVER seen — exactly the real-world use case.

Run analyze_pairs.py first to populate the cache.
"""

import numpy as np
from pathlib import Path

from sluty import core as T
from sluty.core import apply_lut

LUT_SIZE = 33
CACHE_DIR = Path("/tmp/npc_lut_cache")


def rgb_err(a, b):
    return np.linalg.norm(a - b, axis=1) * 255.0


def cell_index(rgb, n):
    r = np.clip(np.round(rgb[:, 0] * (n - 1)), 0, n - 1).astype(np.int64)
    g = np.clip(np.round(rgb[:, 1] * (n - 1)), 0, n - 1).astype(np.int64)
    b = np.clip(np.round(rgb[:, 2] * (n - 1)), 0, n - 1).astype(np.int64)
    return (b * n + g) * n + r


def main():
    imgs = sorted(CACHE_DIR.glob("img_*.npz"))
    data = [np.load(p) for p in imgs]
    n = len(data)
    print(f"Loaded {n} cached images\n")

    counts = np.stack([d["counts"] for d in data])
    ssrc = np.stack([d["ssrc"] for d in data])
    sdst = np.stack([d["sdst"] for d in data])

    print("Leave-one-image-out CV — error reproducing UNSEEN Documentary Chrome photos")
    print(f"{'held-out image':<26}{'floor':>7}{'NN':>8}{'RBF':>8}{'RBF win':>9}")
    print("-" * 58)

    rows = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        c = counts[keep].sum(0)
        ss = ssrc[keep].sum(0)
        sd = sdst[keep].sum(0)

        sflat = data[i]["samp_flat"]
        schrome = data[i]["samp_chrome"]

        # best possible on this image (its own cell mean) = floor
        ci, cs, cd = T.accumulate_cells(sflat, schrome, LUT_SIZE)
        idx = cell_index(sflat, LUT_SIZE)
        floor = rgb_err(np.where(ci[idx, None] > 0, cd[idx] / np.maximum(ci[idx, None], 1), 0),
                        schrome).mean()

        lut_nn = T.fit_lut(c, ss, sd, LUT_SIZE, method="nn", verbose=False)
        lut_rbf = T.fit_lut(c, ss, sd, LUT_SIZE, method="rbf",
                            kernel="thin_plate_spline", verbose=False)
        e_nn = rgb_err(apply_lut(lut_nn, LUT_SIZE, sflat), schrome).mean()
        e_rbf = rgb_err(apply_lut(lut_rbf, LUT_SIZE, sflat), schrome).mean()
        win = 100 * (1 - e_rbf / e_nn)
        rows.append((floor, e_nn, e_rbf, win))
        print(f"{str(data[i]['name']):<26}{floor:>7.2f}{e_nn:>8.2f}{e_rbf:>8.2f}{win:>8.0f}%")

    rows = np.array(rows)
    print("-" * 58)
    print(f"{'MEAN':<26}{rows[:,0].mean():>7.2f}{rows[:,1].mean():>8.2f}"
          f"{rows[:,2].mean():>8.2f}{100*(1-rows[:,2].mean()/rows[:,1].mean()):>8.0f}%")
    print("\nfloor = best any 3D LUT could do on that image (its own cell means)")
    print("NN/RBF = error reproducing the held-out photo from a LUT built on the others")
    print("\nGap between RBF and floor = how much is lost by not having that photo's colours;")
    print("shrinks toward the floor as you add more / more-varied photos.")


if __name__ == "__main__":
    main()
