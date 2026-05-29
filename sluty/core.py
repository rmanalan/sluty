"""
sluty.core — derive a 3D .cube LUT from before/after image pairs.

Given pairs of images that are identical except for a colour/tone transform
(e.g. the same frame exported with and without a film-look preset, a camera
picture control, a Lightroom/Capture One style, or a DaVinci grade), this fits
a 3D LUT that reproduces that transform.

Method ('rbf', default): bin every pixel into the LUT grid to denoise and
compress millions of pixel pairs down to a few thousand representative
(input_colour → output_colour) anchors, then fit a radial-basis-function model
with a degree-1 polynomial tail. The polynomial tail makes the saturated gamut
corners — which no photograph ever fills — extrapolate along a smooth linear
trend instead of the flat nearest-neighbour copy of naïve approaches. High-
sample cells are pinned; low-sample cells relax toward the global trend
(per-anchor smoothing ∝ 1/sample_count). Past `max_anchors` it switches to a
local RBF (k nearest neighbours per query) so multi-image runs stay fast.

A 'nn' method (binning + 3D nearest-neighbour fill) is kept for comparison.
"""

from pathlib import Path
import numpy as np

_IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_image_float32(path: str) -> np.ndarray:
    """Read an image, strip alpha, return float32 RGB in [0, 1].

    Uses tifffile for TIFFs (preserves 16-bit) and Pillow for everything else.
    """
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            img = tifffile.imread(str(p))
        except ImportError as e:
            raise ImportError("Reading TIFFs needs tifffile: pip install tifffile") from e
    else:
        try:
            from PIL import Image
            img = np.asarray(Image.open(p))
        except ImportError as e:
            raise ImportError("Reading non-TIFF images needs Pillow: pip install pillow") from e

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    if np.issubdtype(img.dtype, np.unsignedinteger):
        return img.astype(np.float32) / np.iinfo(img.dtype).max
    elif np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float32)
    raise TypeError(f"Unsupported image dtype: {img.dtype}")


# ---------------------------------------------------------------------------
# Grid + binning
# ---------------------------------------------------------------------------

def cube_grid(lut_size: int) -> np.ndarray:
    """(lut_size³, 3) grid node colours in .cube order: R fastest, B slowest."""
    t = np.linspace(0.0, 1.0, lut_size)
    B, G, R = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1)


def accumulate_cells(src: np.ndarray, dst: np.ndarray, lut_size: int,
                     counts=None, sum_src=None, sum_dst=None):
    """
    Bin pixel pairs into LUT cells. Accumulates into provided arrays (for
    multi-image runs) or fresh ones. Returns (counts, sum_src, sum_dst).
    Cell flat index uses .cube order: flat = b*N² + g*N + r.
    """
    n_cells = lut_size ** 3
    if counts is None:
        counts = np.zeros(n_cells, dtype=np.int64)
        sum_src = np.zeros((n_cells, 3), dtype=np.float64)
        sum_dst = np.zeros((n_cells, 3), dtype=np.float64)

    r = np.clip(np.round(src[:, 0] * (lut_size - 1)).astype(np.int64), 0, lut_size - 1)
    g = np.clip(np.round(src[:, 1] * (lut_size - 1)).astype(np.int64), 0, lut_size - 1)
    b = np.clip(np.round(src[:, 2] * (lut_size - 1)).astype(np.int64), 0, lut_size - 1)
    flat = (b * lut_size + g) * lut_size + r

    counts += np.bincount(flat, minlength=n_cells)
    for ch in range(3):
        sum_src[:, ch] += np.bincount(flat, weights=src[:, ch].astype(np.float64), minlength=n_cells)
        sum_dst[:, ch] += np.bincount(flat, weights=dst[:, ch].astype(np.float64), minlength=n_cells)
    return counts, sum_src, sum_dst


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def enforce_monotonic_lut(lut_flat, lut_size):
    """Force each output channel to be non-decreasing along its own input axis.

    Guards against thin-plate-spline ringing in extrapolated regions (shadows,
    saturated corners) where the fit can reverse direction and produce colored
    blotches/banding. Cube order is R fastest, B slowest → reshape to [b,g,r,3];
    R↔axis 2/chan 0, G↔axis 1/chan 1, B↔axis 0/chan 2.
    """
    lut = np.asarray(lut_flat, dtype=np.float64).reshape(
        lut_size, lut_size, lut_size, 3)
    for axis, chan in ((2, 0), (1, 1), (0, 2)):
        lut[..., chan] = np.maximum.accumulate(lut[..., chan], axis=axis)
    return np.clip(lut.reshape(-1, 3), 0.0, 1.0).astype(np.float32)


def shadow_desaturate_lut(lut_flat, hi):
    """Roll output chroma toward neutral in deep shadows (output luma < ``hi``).

    Low-coverage fits extrapolate the deep-shadow mapping and tend to *amplify*
    chroma there, turning faint shadow noise into coherent coloured blotches.
    Pulling each dark output node toward its own grey removes the blotching while
    leaving midtones/highlights — and thus the look — untouched. ``hi`` is the
    output-luminance threshold (e.g. 0.18); 0 disables. Weight ramps linearly
    from full desaturation at luma 0 to none at ``hi``.
    """
    if not hi:
        return np.asarray(lut_flat, dtype=np.float32)
    lut = np.asarray(lut_flat, dtype=np.float64)
    luma = lut @ np.array([0.2126, 0.7152, 0.0722])
    grey = lut.mean(axis=1, keepdims=True)
    w = np.clip((hi - luma) / hi, 0.0, 1.0)[:, None]
    return np.clip(lut * (1.0 - w) + grey * w, 0.0, 1.0).astype(np.float32)


def fit_lut(counts, sum_src, sum_dst, lut_size,
            method="rbf", kernel="thin_plate_spline", smoothing=0.02,
            epsilon=None, max_anchors=4000, verbose=True,
            enforce_monotonic=False, shadow_desat=0.0):
    """Build an (n_cells, 3) LUT from accumulated cell statistics."""
    n_cells = lut_size ** 3
    populated = counts > 0
    n_pop = int(populated.sum())
    if verbose:
        print(f"  Populated cells: {n_pop:,} / {n_cells:,} ({100*n_pop/n_cells:.1f}%)")
    if n_pop == 0:
        raise ValueError("No populated cells — empty input?")

    in_mean = sum_src[populated] / counts[populated, None]
    out_mean = sum_dst[populated] / counts[populated, None]
    pop_counts = counts[populated]
    grid = cube_grid(lut_size)

    if method == "nn":
        from scipy.ndimage import distance_transform_edt
        lut_mean = np.full((n_cells, 3), np.nan)
        lut_mean[populated] = out_mean
        lut_3d = lut_mean.reshape(lut_size, lut_size, lut_size, 3)
        empty_3d = ~populated.reshape(lut_size, lut_size, lut_size)
        if empty_3d.any():
            _, nn = distance_transform_edt(empty_3d, return_indices=True)
            for ch in range(3):
                vol = lut_3d[:, :, :, ch]
                vol[empty_3d] = vol[nn[0][empty_3d], nn[1][empty_3d], nn[2][empty_3d]]
        lut_out = np.clip(lut_3d.reshape(n_cells, 3), 0.0, 1.0).astype(np.float32)
        if enforce_monotonic:
            lut_out = enforce_monotonic_lut(lut_out, lut_size)
        return shadow_desaturate_lut(lut_out, shadow_desat)

    if method == "rbf":
        from scipy.interpolate import RBFInterpolator
        anchors_x = in_mean
        anchors_y = out_mean
        # Per-anchor smoothing: trust high-count cells, relax low-count ones.
        smooth = smoothing / pop_counts.astype(np.float64)

        kw = dict(kernel=kernel, smoothing=smooth, degree=1)  # linear tail → sane corners
        if kernel in ("multiquadric", "inverse_multiquadric",
                      "inverse_quadratic", "gaussian"):
            kw["epsilon"] = epsilon if epsilon is not None else 0.1

        # Global solve is O(P³); past max_anchors use a local RBF so multi-image
        # runs stay fast and use ALL anchors instead of discarding most.
        if len(anchors_x) > max_anchors:
            kw["neighbors"] = min(64, len(anchors_x))
            if verbose:
                print(f"  {len(anchors_x):,} anchors → local RBF (neighbors={kw['neighbors']})")

        rbf = RBFInterpolator(anchors_x, anchors_y, **kw)
        lut = rbf(grid)
        if verbose:
            over = ((lut < -0.02) | (lut > 1.02)).any(1)
            if over.any():
                print(f"  {over.sum():,} nodes ({100*over.mean():.1f}%) extrapolated past "
                      f"[0,1] then clipped (deep saturated corners with no measured data)")
        lut_out = np.clip(lut, 0.0, 1.0).astype(np.float32)
        if enforce_monotonic:
            had = int((np.diff(lut_out.reshape(lut_size, lut_size, lut_size, 3),
                               axis=2)[..., 0] < -1e-4).sum())
            lut_out = enforce_monotonic_lut(lut_out, lut_size)
            if verbose and had:
                print(f"  monotonicity enforced (removed extrapolation ringing "
                      f"in shadows/saturated corners)")
        if shadow_desat:
            lut_out = shadow_desaturate_lut(lut_out, shadow_desat)
            if verbose:
                print(f"  deep-shadow chroma rolled off below luma {shadow_desat:g} "
                      f"(removes extrapolated colour blotching in darks)")
        return lut_out

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Apply + safety
# ---------------------------------------------------------------------------

def apply_lut(lut_flat, lut_size, rgb):
    """Apply a .cube-order (R fast, B slow) LUT to (N,3) RGB via trilinear interp."""
    from scipy.interpolate import RegularGridInterpolator
    t = np.linspace(0, 1, lut_size)
    vol = np.asarray(lut_flat).reshape(lut_size, lut_size, lut_size, 3)
    interp = RegularGridInterpolator((t, t, t), vol, bounds_error=False, fill_value=None)
    pts = np.stack([rgb[:, 2], rgb[:, 1], rgb[:, 0]], axis=1)  # (b, g, r)
    return np.clip(interp(pts), 0, 1)


def validate_gamut_corners(lut_flat, lut_size, verbose=True):
    """
    Sanity-check the extrapolated gamut corners. A colour transform may shift
    hue intentionally, so we only flag GROSS anomalies: a pure primary that
    loses its dominant channel, or a primary that collapses toward neutral.
    Returns a list of warning strings.
    """
    def sample(rgb):
        r, g, b = [int(round(v * (lut_size - 1))) for v in rgb]
        return lut_flat[(b * lut_size + g) * lut_size + r]

    issues = []
    for rgb, name, ch in [([1, 0, 0], "Red", 0), ([0, 1, 0], "Green", 1), ([0, 0, 1], "Blue", 2)]:
        out = sample(rgb)
        if np.argmax(out) != ch:
            issues.append(f"{name} corner lost dominant channel → {np.round(out, 3)}")
        if out.max() - out.min() < 0.10:
            issues.append(f"{name} corner collapsed toward neutral → {np.round(out, 3)}")
    if verbose:
        if issues:
            for s in issues:
                print(f"  ⚠ gamut corner: {s}")
            print("    (corners are extrapolated — add a colourful frame to measure them directly)")
        else:
            print("  gamut corners OK (primaries keep their dominant channel, no neutral collapse)")
    return issues


# ---------------------------------------------------------------------------
# Output + high-level driver
# ---------------------------------------------------------------------------

def write_cube(lut_out, cube_path, lut_size, header=""):
    cube_path_obj = Path(cube_path)
    cube_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with cube_path_obj.open("w") as f:
        f.write("# Generated by sLUTy (https://github.com — sluty)\n")
        if header:
            f.write(f"# {header}\n")
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for r, g, b in lut_out:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def pairs_from_dirs(source_dir: str, target_dir: str):
    """Match images by identical filename across two directories. Returns pairs."""
    sd, td = Path(source_dir), Path(target_dir)
    srcs = {p.name: p for p in sd.iterdir() if p.suffix.lower() in _IMAGE_EXTS}
    tgts = {p.name: p for p in td.iterdir() if p.suffix.lower() in _IMAGE_EXTS}
    common = sorted(set(srcs) & set(tgts))
    only_src = sorted(set(srcs) - set(tgts))
    only_tgt = sorted(set(tgts) - set(srcs))
    if only_src:
        print(f"  ! {len(only_src)} file(s) only in source dir, skipped: {only_src[:3]}…")
    if only_tgt:
        print(f"  ! {len(only_tgt)} file(s) only in target dir, skipped: {only_tgt[:3]}…")
    if not common:
        raise ValueError("No matching filenames between the two directories.")
    return [(str(srcs[n]), str(tgts[n])) for n in common]


def derive_lut(pairs, cube_path, lut_size=33, method="rbf",
               kernel="thin_plate_spline", smoothing=0.02,
               enforce_monotonic=False, shadow_desat=0.0):
    """Read pairs, accumulate, fit, validate, and write a .cube file."""
    counts = sum_src = sum_dst = None
    total_px = 0
    for idx, (src_path, tgt_path) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {Path(src_path).name} → {Path(tgt_path).name}")
        img_src = read_image_float32(src_path)
        img_tgt = read_image_float32(tgt_path)
        if img_src.shape != img_tgt.shape:
            raise ValueError(f"Shape mismatch in pair {idx}: {img_src.shape} vs {img_tgt.shape}")
        N = img_src.shape[0] * img_src.shape[1]
        total_px += N
        counts, sum_src, sum_dst = accumulate_cells(
            img_src.reshape(N, 3), img_tgt.reshape(N, 3), lut_size,
            counts, sum_src, sum_dst,
        )

    print(f"\nTotal pixels binned: {total_px:,}")
    print(f"Fitting LUT (method={method}, kernel={kernel}) …")
    lut_out = fit_lut(counts, sum_src, sum_dst, lut_size, method, kernel, smoothing,
                      enforce_monotonic=enforce_monotonic, shadow_desat=shadow_desat)

    cov = 100 * (counts > 0).sum() / lut_size ** 3
    if method == "rbf":
        validate_gamut_corners(lut_out, lut_size)

    pair_names = ", ".join(f"{Path(a).name}→{Path(b).name}" for a, b in pairs)
    mono_note = ("\n# Monotonicity enforced (per-axis) to suppress extrapolation ringing."
                 if enforce_monotonic else "")
    desat_note = (f"\n# Deep-shadow chroma rolled off below luma {shadow_desat:g} "
                  f"(removes extrapolated colour blotching in darks)."
                  if shadow_desat else "")
    header = (f"Sources: {pair_names}\n"
              f"# Coverage: {cov:.1f}% of the {lut_size}³ grid measured directly; "
              f"the rest is extrapolated (least reliable in saturated corners).\n"
              f"# Method: {method} ({kernel}); {len(pairs)} pair(s), {total_px:,} pixels."
              f"{mono_note}{desat_note}")
    write_cube(lut_out, cube_path, lut_size, header)
    print(f"Done — {lut_size}³ LUT written to {cube_path}")
    return lut_out
