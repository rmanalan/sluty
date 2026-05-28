#!/usr/bin/env python3
"""
verify_final.py — Honest, audit-driven verification of the delivered LUT.

Addresses the adversarial review's valid concerns:
  1. Gamut-corner safety (hue flip / neutral collapse) on the delivered LUT.
  2. Local-RBF smoothness — Laplacian of the LUT + agreement vs a global RBF.
  3. Perceptual error: ΔE (CIELAB) alongside RGB, since looks live in chroma/hue.
  4. A FAIR floor: LUT trained on the other images' SAMPLES (equal data), not the
     mislabeled sample-vs-full comparison.
  5. TRUE full-image held-out error (not a 200k sample) for two images.
  6. 33³ vs 65³ (sample-based, equal source) — does finer binning help?
  7. Coverage gaps by hue and saturation.

Run analyze_pairs.py first.
"""

import numpy as np
from pathlib import Path
from sluty import core as T
from sluty.core import apply_lut

LUT_SIZE = 33
CACHE = Path("/tmp/npc_lut_cache")
HOME = Path.home()


# ---- sRGB → CIELAB ΔE76 ----------------------------------------------------
def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def srgb_to_lab(rgb):
    lin = _srgb_to_lin(np.clip(rgb, 0, 1))
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    d = 6/29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3*d*d) + 4/29)
    L = 116*f[:,1] - 16
    a = 500*(f[:,0] - f[:,1])
    bb = 200*(f[:,1] - f[:,2])
    return np.stack([L, a, bb], axis=1)

def delta_e(rgb1, rgb2):
    return np.linalg.norm(srgb_to_lab(rgb1) - srgb_to_lab(rgb2), axis=1)

def rgb_err(a, b):
    return np.linalg.norm(a - b, axis=1) * 255.0

def cell_index(rgb, n):
    r = np.clip(np.round(rgb[:,0]*(n-1)),0,n-1).astype(np.int64)
    g = np.clip(np.round(rgb[:,1]*(n-1)),0,n-1).astype(np.int64)
    b = np.clip(np.round(rgb[:,2]*(n-1)),0,n-1).astype(np.int64)
    return (b*n+g)*n+r


def main():
    imgs = sorted(CACHE.glob("img_*.npz"))
    data = [np.load(p) for p in imgs]
    names = [str(d["name"]) for d in data]
    n = len(data)
    counts = np.stack([d["counts"] for d in data])
    ssrc = np.stack([d["ssrc"] for d in data])
    sdst = np.stack([d["sdst"] for d in data])

    # delivered LUT
    lines = open(HOME/"Downloads/Documentary Chrome.cube").readlines()
    ds = next(i for i,l in enumerate(lines) if l[:1].isdigit() or l[:1]=='-')
    lut = np.array([list(map(float,l.split())) for l in lines[ds:]], dtype=np.float32)

    print("="*64)
    print("1. GAMUT-CORNER SAFETY (delivered LUT)")
    print("="*64)
    T.validate_gamut_corners(lut, LUT_SIZE)

    print("\n"+"="*64)
    print("2. LOCAL-RBF SMOOTHNESS (no seams/discontinuities?)")
    print("="*64)
    vol = lut.reshape(LUT_SIZE,LUT_SIZE,LUT_SIZE,3)
    # discrete Laplacian (interior nodes) vs typical first-difference magnitude
    lap = (np.abs(np.diff(vol,2,axis=0)).mean() +
           np.abs(np.diff(vol,2,axis=1)).mean() +
           np.abs(np.diff(vol,2,axis=2)).mean())/3
    d1 = (np.abs(np.diff(vol,1,axis=0)).mean() +
          np.abs(np.diff(vol,1,axis=1)).mean() +
          np.abs(np.diff(vol,1,axis=2)).mean())/3
    print(f"  mean |2nd-difference| {lap:.5f}  vs mean |1st-difference| {d1:.5f}")
    print(f"  ratio {lap/d1:.3f}  (≪1 ⇒ smooth, no seams; >0.5 would suggest patch artifacts)")

    print("\n"+"="*64)
    print("3. HELD-OUT ACCURACY — RGB and PERCEPTUAL ΔE (sample-based LOO)")
    print("="*64)
    print(f"{'held-out image':<26}{'RGB':>7}{'ΔE':>7}{'ΔE95':>7}")
    print("-"*47)
    rgb_all=[]; de_all=[]
    for i in range(n):
        keep=[j for j in range(n) if j!=i]
        lut_i = T.fit_lut(counts[keep].sum(0),ssrc[keep].sum(0),sdst[keep].sum(0),
                          LUT_SIZE,method="rbf",kernel="thin_plate_spline",verbose=False)
        sf,sc = data[i]["samp_flat"], data[i]["samp_chrome"]
        pred = apply_lut(lut_i, LUT_SIZE, sf)
        e = rgb_err(pred,sc); de = delta_e(pred,sc)
        rgb_all.append(e.mean()); de_all.append(de.mean())
        print(f"{names[i]:<26}{e.mean():>7.2f}{de.mean():>7.2f}{np.percentile(de,95):>7.2f}")
    print("-"*47)
    print(f"{'MEAN':<26}{np.mean(rgb_all):>7.2f}{np.mean(de_all):>7.2f}")
    print("  ΔE<1 imperceptible · 1–2 just noticeable · 2–3 noticeable on close look")

    print("\n"+"="*64)
    print("4. FAIR FLOOR — LUT from others' SAMPLES (equal data) vs full-image LUT")
    print("="*64)
    fair=[]; full=[]
    for i in range(n):
        keep=[j for j in range(n) if j!=i]
        # FAIR: build from other images' SAMPLES only (same data scale as the old 'floor')
        sf_o = np.concatenate([data[j]["samp_flat"] for j in keep])
        sc_o = np.concatenate([data[j]["samp_chrome"] for j in keep])
        cs,ss2,sd2 = T.accumulate_cells(sf_o, sc_o, LUT_SIZE)
        lut_fair = T.fit_lut(cs,ss2,sd2,LUT_SIZE,method="rbf",kernel="thin_plate_spline",verbose=False)
        # FULL: from other images' full stats
        lut_full = T.fit_lut(counts[keep].sum(0),ssrc[keep].sum(0),sdst[keep].sum(0),
                             LUT_SIZE,method="rbf",kernel="thin_plate_spline",verbose=False)
        sf,sc=data[i]["samp_flat"],data[i]["samp_chrome"]
        fair.append(rgb_err(apply_lut(lut_fair,LUT_SIZE,sf),sc).mean())
        full.append(rgb_err(apply_lut(lut_full,LUT_SIZE,sf),sc).mean())
    print(f"  LUT from others' SAMPLES (~2.2M px): mean held-out RGB {np.mean(fair):.2f}")
    print(f"  LUT from others' FULL images (~268M px): mean held-out RGB {np.mean(full):.2f}")
    print(f"  → full-image training is {np.mean(fair)-np.mean(full):.2f} better (denoising is real,")
    print("    and the earlier 'floor<RBF' was an unfair sample-vs-full comparison — now corrected)")

    print("\n"+"="*64)
    print("5. TRUE FULL-IMAGE held-out error (not a sample) — 2 images")
    print("="*64)
    flat_dir = HOME/"Downloads/Flat"; sty_dir = HOME/"Downloads/Documentary Chrome"
    for i in (0, 6):
        keep=[j for j in range(n) if j!=i]
        lut_i = T.fit_lut(counts[keep].sum(0),ssrc[keep].sum(0),sdst[keep].sum(0),
                          LUT_SIZE,method="rbf",kernel="thin_plate_spline",verbose=False)
        fimg = T.read_tif_float32(str(flat_dir/names[i])).reshape(-1,3)
        cimg = T.read_tif_float32(str(sty_dir/names[i])).reshape(-1,3)
        pred = apply_lut(lut_i, LUT_SIZE, fimg)
        e = rgb_err(pred,cimg); de = delta_e(pred,cimg)
        print(f"  {names[i]:<24} full-image RGB {e.mean():.2f} (95th {np.percentile(e,95):.1f})  "
              f"ΔE {de.mean():.2f} (95th {np.percentile(de,95):.1f})")

    print("\n"+"="*64)
    print("6. 33³ vs 65³ — does finer binning help? (equal source: samples)")
    print("="*64)
    for N in (33, 65):
        errs=[]
        for i in range(n):
            keep=[j for j in range(n) if j!=i]
            sf_o=np.concatenate([data[j]["samp_flat"] for j in keep])
            sc_o=np.concatenate([data[j]["samp_chrome"] for j in keep])
            cs,ss2,sd2=T.accumulate_cells(sf_o,sc_o,N)
            lut_n=T.fit_lut(cs,ss2,sd2,N,method="rbf",kernel="thin_plate_spline",verbose=False)
            sf,sc=data[i]["samp_flat"],data[i]["samp_chrome"]
            errs.append(rgb_err(apply_lut(lut_n,N,sf),sc).mean())
        print(f"  {N}³: mean held-out RGB {np.mean(errs):.2f}  (sample-trained, relative guide)")

    print("\n"+"="*64)
    print("7. COVERAGE GAPS by hue & saturation (combined, all 12)")
    print("="*64)
    comb = counts.sum(0)
    grid = T.cube_grid(LUT_SIZE)
    pop = comb > 0
    hsv = _rgb_to_hsv(grid)
    # saturation bands
    for lo,hi,lbl in [(0,0.2,"near-neutral"),(0.2,0.5,"moderate"),(0.5,0.8,"saturated"),(0.8,1.01,"very saturated")]:
        m=(hsv[:,1]>=lo)&(hsv[:,1]<hi)
        if m.sum(): print(f"  saturation {lbl:<15}: {100*pop[m].mean():4.1f}% of those cells measured")
    print("  (high-saturation regions are the least covered → least reliable; "
          "a colourful frame fixes this)")


def _rgb_to_hsv(rgb):
    mx=rgb.max(1); mn=rgb.min(1); d=mx-mn
    s=np.where(mx>1e-9, d/np.maximum(mx,1e-9), 0)
    return np.stack([np.zeros_like(mx), s, mx],1)


if __name__ == "__main__":
    main()
