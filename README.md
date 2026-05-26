# npc-to-lut

Convert Nikon Custom Picture Control files (`.NP3` / `.NCP`) into standard 3D `.cube` LUTs for DaVinci Resolve and Premiere Pro — no camera, NEF file, or GUI software required.

## Supported formats

| Extension | Cameras |
|-----------|---------|
| `.NP3` | Nikon Z-series (Zf, Z8, Z9, Z6 III …) — Flexible Color Picture Control |
| `.NCP` | D-series and older Z-series — Classic Custom Picture Control |

## Installation

```bash
pip install -e .
```

Requires Python 3.9+. Dependencies (`numpy`, `scipy`, `click`) all have pre-built ARM wheels, so installation works on a Raspberry Pi without compilation.

## Usage

### Convert a Picture Control to a .cube LUT

```bash
npc-to-lut convert MyPreset.NP3 -o MyPreset.cube
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | `<input>.cube` | Output path |
| `--lut-size` | `33` | Grid size — 17, 33, or 65 |
| `--debug` | off | Print per-stage delta vs. identity |

### Inspect all parsed parameters

```bash
npc-to-lut info MyPreset.NP3
```

Outputs a JSON document with every decoded field: contrast, saturation, highlights, shadows, tone curve presence, all 8 color blender bands, 3-way color grading, etc.

### Validate a .cube file

```bash
npc-to-lut validate MyPreset.cube
```

Checks that black/white corners are preserved and reports the maximum deviation from an identity LUT.

## How it works

### What gets baked into the LUT

| Parameter | Included | Notes |
|-----------|----------|-------|
| Tone curve | ✅ | 257-entry 1D LUT or cubic spline |
| Contrast | ✅ | Parametric sigmoid around mid-gray |
| Highlights / Shadows | ✅ | Luminance-masked zone curves |
| White / Black level | ✅ | Output range scaling |
| Saturation | ✅ | HSL chroma scale |
| Color Blender | ✅ | 8-band per-hue Gaussian-weighted HSL shifts |
| Color Grading | ✅ | 3-way color wheel (shadows / midtone / highlights) |
| Sharpening / Clarity | ❌ | Spatial filters — not a color transform |

### Known limitation — base Tonal Response Curve

Nikon's predefined base curves (Standard, Neutral, Vivid) are embedded in camera firmware and are **not stored** in the Picture Control file. The generated LUT represents only the **user-defined adjustments** layered on top of the base curve. For the most accurate result, use a base Picture Control set to **Neutral** before applying your custom controls.

### Processing pipeline

```
tone curve → contrast → highlights/shadows → white/black level
→ saturation → color blender → color grading
```

The 33×33×33 identity grid (35,937 RGB nodes) is passed through each transform in sequence. All math runs in sRGB-encoded [0, 1] space, matching how the camera applies Picture Controls post-gamma.

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## File format references

- **NP3 binary layout**: [ssssota/nikon-flexible-color-picture-control](https://github.com/ssssota/nikon-flexible-color-picture-control)
- **NCP binary layout**: [horshack-dpreview/NikonPictureControlsDev](https://github.com/horshack-dpreview/NikonPictureControlsDev)
- **Adobe .cube spec**: Adobe Cube LUT Specification 1.0
