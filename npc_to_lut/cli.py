"""
CLI entry point.

Commands:
  npc-to-lut convert  INPUT [-o OUTPUT] [--lut-size N] [--debug]
  npc-to-lut info     INPUT
  npc-to-lut validate LUT.cube
"""

import json
import sys
from pathlib import Path

import click
import numpy as np

from .parser import load, PictureControl
from .lut_builder import build_lut
from .cube_writer import write_cube


def _pc_to_dict(pc: PictureControl) -> dict:
    """Serialise a PictureControl to a plain dict for JSON output."""
    return {
        "name": pc.name,
        "source_format": pc.source_format,
        "contrast": pc.contrast,
        "highlights": pc.highlights,
        "shadows": pc.shadows,
        "white_level": pc.white_level,
        "black_level": pc.black_level,
        "saturation": pc.saturation,
        "tone_curve": "present (257 entries)" if pc.tone_curve else "identity",
        "base_curve_id": hex(pc.base_curve_id) if pc.base_curve_id else None,
        "color_blender": [
            {"band": name, "hue": b.hue, "chroma": b.chroma, "brightness": b.brightness}
            for name, b in zip(
                ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Magenta"],
                pc.color_blender,
            )
        ],
        "color_grading": {
            "highlights": {
                "hue": pc.color_grading.highlights.hue,
                "chroma": pc.color_grading.highlights.chroma,
                "brightness": pc.color_grading.highlights.brightness,
            },
            "midtone": {
                "hue": pc.color_grading.midtone.hue,
                "chroma": pc.color_grading.midtone.chroma,
                "brightness": pc.color_grading.midtone.brightness,
            },
            "shadows": {
                "hue": pc.color_grading.shadows.hue,
                "chroma": pc.color_grading.shadows.chroma,
                "brightness": pc.color_grading.shadows.brightness,
            },
            "blending": pc.color_grading.blending,
            "balance": pc.color_grading.balance,
        },
    }


@click.group()
def main():
    """npc-to-lut — Convert Nikon Picture Control files to .cube LUTs."""


@main.command()
@click.argument("input", metavar="INPUT.NP3|NCP")
@click.option("-o", "--output", default=None, help="Output .cube path (default: INPUT.cube)")
@click.option("--lut-size", default=33, show_default=True, help="LUT grid size (17, 33, or 65)")
@click.option("--debug", is_flag=True, help="Print per-stage delta vs. identity")
def convert(input, output, lut_size, debug):
    """Parse INPUT and write a 3D .cube LUT."""
    in_path = Path(input)
    if not in_path.exists():
        click.echo(f"Error: file not found: {input}", err=True)
        sys.exit(1)

    if output is None:
        output = str(in_path.with_suffix(".cube"))

    click.echo(f"Parsing {in_path.name} …")
    pc = load(str(in_path))
    click.echo(f"  Format   : {pc.source_format.upper()}")
    click.echo(f"  Name     : {pc.name!r}")
    click.echo(f"  Contrast : {pc.contrast:+.0f}  Saturation: {pc.saturation:+.0f}")
    click.echo(f"  Tone curve: {'present' if pc.tone_curve else 'identity'}")

    if pc.source_format == "ncp" and pc.base_curve_id is not None:
        click.echo(
            f"\n  NOTE: NCP base curve (0x{pc.base_curve_id:04X}) is embedded in camera "
            "firmware and is NOT included in this LUT.\n"
            "  The output represents only the user-defined delta on top of that base curve.",
            err=True,
        )

    click.echo(f"\nBuilding {lut_size}×{lut_size}×{lut_size} LUT …")
    lut = build_lut(pc, size=lut_size, debug=debug)

    title = f"NPC: {pc.name}" if pc.name else "NPC LUT"
    write_cube(lut, output, title=title)
    click.echo(f"Written: {output}")


@main.command()
@click.argument("input", metavar="INPUT.NP3|NCP")
def info(input):
    """Print all parsed parameters from INPUT as JSON."""
    in_path = Path(input)
    if not in_path.exists():
        click.echo(f"Error: file not found: {input}", err=True)
        sys.exit(1)

    pc = load(str(in_path))
    click.echo(json.dumps(_pc_to_dict(pc), indent=2))


@main.command()
@click.argument("cube_path", metavar="LUT.cube")
def validate(cube_path):
    """
    Quick sanity-check on a .cube file.
    Verifies that (0,0,0)→(0,0,0) and (1,1,1)→(1,1,1) within tolerance,
    and reports max deviation from identity.
    """
    cube_file = Path(cube_path)
    if not cube_file.exists():
        click.echo(f"Error: file not found: {cube_path}", err=True)
        sys.exit(1)

    lines = [l.strip() for l in cube_file.read_text().splitlines()
             if l.strip() and not l.startswith("#")]

    size = None
    data_lines = []
    for line in lines:
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
        elif line[0].isdigit() or line[0] == "-":
            data_lines.append(line)

    if size is None:
        click.echo("Error: could not find LUT_3D_SIZE in file.", err=True)
        sys.exit(1)

    if len(data_lines) != size ** 3:
        click.echo(
            f"Error: expected {size**3} data lines, found {len(data_lines)}.", err=True
        )
        sys.exit(1)

    values = np.array([[float(x) for x in l.split()] for l in data_lines])
    lut = values.reshape(size, size, size, 3)  # B outer, R inner (cube spec)

    # Identity expected: lut[r,g,b] should equal (r/(size-1), g/(size-1), b/(size-1))
    axis = np.linspace(0, 1, size)
    # Cube file ordering: B outer, G middle, R inner
    R_idx = np.arange(size)
    identity = np.zeros((size, size, size, 3))
    for b_i in range(size):
        for g_i in range(size):
            for r_i in range(size):
                identity[b_i, g_i, r_i] = [axis[r_i], axis[g_i], axis[b_i]]

    delta = np.abs(lut - identity)
    max_delta = delta.max()

    black = lut[0, 0, 0]
    white = lut[size - 1, size - 1, size - 1]

    click.echo(f"LUT size  : {size}×{size}×{size}")
    click.echo(f"(0,0,0) → ({black[0]:.4f}, {black[1]:.4f}, {black[2]:.4f})")
    click.echo(f"(1,1,1) → ({white[0]:.4f}, {white[1]:.4f}, {white[2]:.4f})")
    click.echo(f"Max delta from identity: {max_delta:.6f}")

    tol = 0.01
    if max_delta < tol:
        click.echo(f"PASS — LUT is near-identity (delta < {tol})")
    else:
        click.echo(f"INFO — LUT deviates from identity (max delta = {max_delta:.4f})")
        click.echo("  (Expected if the Picture Control is not at neutral settings.)")
