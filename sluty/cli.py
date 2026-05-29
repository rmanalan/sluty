"""sLUTy command-line interface."""

import argparse
from .core import pairs_from_dirs, derive_lut


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="sluty",
        description="Derive a 3D .cube LUT from one or more before/after image pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "A pair is the SAME frame exported twice: a 'source' (neutral/flat) and a\n"
            "'target' (the look you want — a preset, picture control, film sim, grade).\n\n"
            "Examples:\n"
            "  sluty source.tif target.tif look.cube\n"
            "  sluty s1.tif t1.tif s2.tif t2.tif look.cube --size 65\n"
            "  sluty --source-dir Flat --target-dir 'Documentary Chrome' look.cube --size 65\n\n"
            "Tip: export pairs with sharpening / clarity / local-contrast OFF — those are\n"
            "spatial effects a per-pixel 3D LUT cannot represent, and leaving them on\n"
            "raises the error floor. Add a few colourful frames to cover saturated hues."
        ),
    )
    ap.add_argument("images_and_cube", nargs="*",
                    help="Interleaved source/target image paths, then the output .cube path")
    ap.add_argument("--source-dir", help="Directory of source images (pair by filename)")
    ap.add_argument("--target-dir", help="Directory of target images (pair by filename)")
    ap.add_argument("--out", help="Output .cube path (required with --source-dir/--target-dir)")
    ap.add_argument("--size", type=int, default=33, metavar="N",
                    help="LUT grid size, N³ entries (default 33; 65 is sharper)")
    ap.add_argument("--method", choices=["rbf", "nn"], default="rbf",
                    help="Fit strategy (default rbf; nn = nearest-neighbour fill)")
    ap.add_argument("--kernel", default="thin_plate_spline", help="RBF kernel")
    ap.add_argument("--smoothing", type=float, default=0.02,
                    help="Base smoothing for a single-sample anchor")
    ap.add_argument("--enforce-monotonic", action="store_true",
                    help="Force each output channel non-decreasing along its input axis. "
                         "Suppresses thin-plate-spline ringing in extrapolated regions "
                         "(shadow blotches, saturated-corner banding) when coverage is low.")
    ap.add_argument("--shadow-desat", type=float, default=0.0, metavar="LUMA",
                    help="Roll deep-shadow chroma toward neutral below this output luma "
                         "(e.g. 0.18; 0 = off). Removes coloured blotching that low-coverage "
                         "fits amplify in the darks; leaves midtones/highlights untouched.")
    args = ap.parse_args()

    if args.source_dir or args.target_dir:
        if not (args.source_dir and args.target_dir and args.out):
            ap.error("--source-dir, --target-dir and --out must be given together.")
        print(f"Matching pairs: {args.source_dir!r}  ↔  {args.target_dir!r}")
        pairs = pairs_from_dirs(args.source_dir, args.target_dir)
        print(f"  {len(pairs)} matched pair(s)")
        cube_path = args.out
    else:
        if len(args.images_and_cube) < 3:
            ap.error("Give interleaved source/target images + output .cube, "
                     "or use --source-dir/--target-dir/--out.")
        cube_path = args.images_and_cube[-1]
        img_paths = args.images_and_cube[:-1]
        if len(img_paths) % 2 != 0:
            ap.error(f"Expected an even number of image paths before the .cube, got {len(img_paths)}.")
        pairs = [(img_paths[i], img_paths[i + 1]) for i in range(0, len(img_paths), 2)]

    derive_lut(pairs, cube_path, args.size, args.method, args.kernel, args.smoothing,
               enforce_monotonic=args.enforce_monotonic, shadow_desat=args.shadow_desat)


if __name__ == "__main__":
    main()
