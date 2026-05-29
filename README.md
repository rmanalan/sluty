# sLUTy

Derive an accurate 3D `.cube` LUT from **before/after image pairs** — no proprietary format parsing, no reverse-engineering, no camera SDK.

If you can produce the same frame twice — once neutral ("source") and once with a look applied ("target") — sLUTy fits a 3D LUT that reproduces that look. It doesn't care *how* the look was made:

- Camera picture controls / film simulations (Nikon Picture Control, Fuji film sim, …) rendered from RAW
- Lightroom / Capture One presets and profiles
- DaVinci Resolve / Premiere grades
- Any plugin or app that transforms colour and tone per-pixel

This is the empirical, measurement-based approach: instead of trying to re-implement someone's proprietary algorithm, it *measures* the transform directly from pixels and reconstructs it. On real data it reproduces the target look to **ΔE ≈ 0.5 (perceptually identical)** on images it never trained on — see [Accuracy](#accuracy).

## Install

```bash
pip install -e .
```

Python 3.9+. Dependencies: `numpy`, `scipy`, `tifffile`, `pillow`.

## How to make a good pair

A *pair* is the **same frame** exported twice from the same source (ideally a RAW), differing only in the colour/tone look:

1. **source** — neutral/flat rendering (e.g. a Flat or Neutral profile)
2. **target** — the look you want to capture

Two things decide how faithful the LUT can be:

- **Turn off spatial effects** (sharpening, clarity/texture, local contrast, noise reduction, lens corrections) on **both** exports. These are neighbourhood-dependent — a per-pixel 3D LUT physically cannot represent them, and leaving them on raises the error floor. In testing, disabling them dropped the irreducible error floor ~4× (14 → 3.5 in 0–255 RGB units).
- **Cover the colours you care about.** One photo fills only a small slice of all possible colours; the rest is extrapolated. Add a few **colourful frames** (saturated subjects, skin, foliage, sky) — coverage stacks across images and the LUT converges to the true look.

Export 16-bit TIFF if you can (cleaner), but PNG/JPEG work too.

## Usage

Single pair:

```bash
sluty source.tif target.tif look.cube
```

Several pairs (interleaved source/target), finer grid:

```bash
sluty s1.tif t1.tif  s2.tif t2.tif  s3.tif t3.tif  look.cube --size 65
```

Whole folders (matched by identical filename — the easiest way to use many pairs):

```bash
sluty --source-dir Flat --target-dir "My Look" --out look.cube --size 65
```

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | `33` | LUT grid size, N³ entries. `65` is measurably sharper; `33` is smaller and plenty for most looks. |
| `--method` | `rbf` | `rbf` (recommended) or `nn` (nearest-neighbour fill, for comparison). |
| `--kernel` | `thin_plate_spline` | RBF kernel — the most robust across coverage levels. |
| `--smoothing` | `0.02` | Base smoothing for a single-sample anchor. |
| `--enforce-monotonic` | off | Force each output channel non-decreasing along its input axis. Suppresses thin-plate-spline ringing in extrapolated regions (shadow blotches, saturated-corner banding) when coverage is low. |
| `--shadow-desat` | `0` (off) | Roll deep-shadow chroma toward neutral below this output luma (e.g. `0.18`). Removes coloured blotching that low-coverage fits amplify in the darks; leaves midtones/highlights — and the look — untouched. |

The generated `.cube` header records how much of the grid was measured directly vs. extrapolated.

## How it works

1. **Bin** every pixel pair into the LUT grid. This denoises (averaging many pixels per cell) and compresses millions of pairs into a few thousand representative `input → output` anchors.
2. **Fit** a radial-basis-function model (`scipy` `RBFInterpolator`, thin-plate-spline kernel, degree-1 polynomial tail) mapping input colour to output colour.
   - The **polynomial tail** makes the saturated gamut corners — which no photograph fills — extrapolate along a smooth linear trend instead of the flat nearest-neighbour copy naïve methods use.
   - **Per-anchor smoothing ∝ 1/sample_count** pins well-measured cells and relaxes sparse, noisy ones toward the global trend.
   - Above ~4000 anchors (multi-image runs) it switches to a **local RBF** (k-nearest-neighbour) so it stays fast and uses every anchor.
3. **Evaluate** the model on the full N³ grid, sanity-check the gamut corners, and write the `.cube`.

All maths runs in the images' encoded (gamma) space, matching how looks are normally applied.

## Accuracy

Validated two ways:

- **Synthetic ground truth** (`validation/validate_lut.py`): against a known transform, RBF beats nearest-neighbour fill by **69–82%** at realistic single-photo coverage, including on a deliberately LUT-unfriendly (piecewise, hue-banded) transform.
- **Real data, leave-one-image-out cross-validation** (`validation/cv_pairs.py`): fitting on a set of pairs and reproducing a *held-out* photo the LUT never saw gives mean **RGB ≈ 1.5 / ΔE ≈ 0.5** (below the threshold of human perception). The cross-validation was independently audited for data leakage.

A key honest finding: the residual error is **coverage-bound, not noise** — beyond a couple million pixels, more pixels of the *same* scenes don't help; only *new colours* do.

## Limitations

- **Spatial effects can't be captured.** Sharpening, clarity, local contrast, halation, grain — anything that depends on neighbouring pixels — is outside what a 3D LUT can represent. Disable them when exporting.
- **Unmeasured colours are extrapolated.** Deep saturated primaries that no source image contains are the least reliable region. Shoot one bold-primary frame to fix it.
- **Bit-exact is not achievable** for proprietary RAW pipelines with consumer tools (no open RAW encoder, non-invertible demosaic, gamut-limited displays). The empirical pair method is the practical optimum — and gets perceptually indistinguishable.

## Validation scripts

`validation/` contains the reproducible verification used to develop and trust the method:

| Script | Purpose |
|--------|---------|
| `validate_lut.py` | Synthetic ground-truth benchmark: RBF vs. nearest-neighbour across coverage levels. |
| `analyze_pairs.py` | Pre-reduce a folder of pairs once; report coverage and the per-image error floor. |
| `cv_pairs.py` | Leave-one-image-out cross-validation (generalisation to unseen photos). |
| `verify_final.py` | Perceptual ΔE, fair-floor, 33³-vs-65³, gamut-corner and coverage audits. |

(The folder-based scripts expect a dataset and write a cache to `/tmp`; `validate_lut.py` is fully self-contained.)

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT.
