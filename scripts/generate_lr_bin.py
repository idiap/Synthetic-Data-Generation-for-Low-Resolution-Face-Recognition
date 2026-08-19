#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: generate_lr_bin.py
#
"""
Generate LR-degraded versions of .bin verification files.

The .bin format stores (bins, issame_list) where bins is a flat list of raw
JPEG bytes: indices 2i and 2i+1 form pair i, labelled by issame_list[i].

Interpolation choices match the training preprocessing pipeline (OpenCV):
    area   -> cv2.INTER_AREA    (recommended for downsampling, avoids aliasing)
    cubic  -> cv2.INTER_CUBIC   (high-quality, may overshoot)
    linear -> cv2.INTER_LINEAR  (fast bilinear)

Auto-naming convention (when --output is not given):
  Upsample-back mode (use --size):
    mode=both   -> <stem>_<size>_<down>_<up>_lr2lr.bin
    mode=second -> <stem>_<size>_<down>_<up>_hr2lr.bin   (pair[0]=HR, pair[1]=LR)
    mode=first  -> <stem>_<size>_<down>_<up>_lr2hr.bin   (pair[0]=LR, pair[1]=HR)
  No-upsample mode (use --output-size):
    --method downsample -> <stem>_noup_<output_size>_<down>_<mode_suffix>.bin
    --method esrgan     -> <stem>_noup_<output_size>_esrgan_<mode_suffix>.bin

Examples:
    # LR2LR: both images degraded to 28x28 (area down, cubic up)
    python scripts/generate_lr_bin.py \\
        --bin data/webface4m/lfw.bin \\
        --size 28 --downsample area --upsample cubic --mode both

    # HR2LR: only the second image of each pair degraded
    python scripts/generate_lr_bin.py \\
        --bin data/webface4m/lfw.bin \\
        --size 28 --downsample area --upsample cubic --mode second

    # Batch: all .bin files in a directory, one combination
    python scripts/generate_lr_bin.py \\
        --bin-dir data/webface4m \\
        --size 14 --downsample area --upsample cubic --mode both

    # Native 16x16 (no upsample-back), bicubic-only
    python scripts/generate_lr_bin.py \\
        --bin data/webface4m/lfw.bin \\
        --output-size 16 --downsample cubic --mode both

    # Native 16x16 with Real-ESRGAN second-order degradation
    python scripts/generate_lr_bin.py \\
        --bin data/webface4m/lfw.bin \\
        --method esrgan --output-size 16 --mode both
"""

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Local import so degradations.py is found whether the script is run from
# repo root or from inside scripts/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


INTERP = {
    "area":   cv2.INTER_AREA,
    "cubic":  cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
}

MODE_SUFFIX = {
    "both":   "lr2lr",
    "second": "hr2lr",
    "first":  "lr2hr",
}


def load_bin(bin_path: Path):
    try:
        with open(bin_path, "rb") as f:
            bins, issame_list = pickle.load(f)
    except UnicodeDecodeError:
        with open(bin_path, "rb") as f:
            bins, issame_list = pickle.load(f, encoding="bytes")
    return bins, issame_list


def degrade(raw_bytes: bytes, args, rng=None) -> bytes:
    """Decode JPEG → degrade → re-encode JPEG.

    Branches on args.method and args.output_size:
      - method='downsample', output_size unset: legacy down-then-up to original H×W
      - method='downsample', output_size set:   downsample only, store at output_size
      - method='esrgan',     output_size set:   Real-ESRGAN second-order, store at output_size
    """
    img_rgb = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
    h, w = img_rgb.shape[:2]

    if args.method == 'esrgan':
        from degradations import realesrgan_degradation
        # The pipeline uses cv2 ops; convert to BGR so per-image gray
        # conversion (used inside Poisson noise) sees the right channel order.
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = realesrgan_degradation(img_bgr, args.output_size, rng)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    else:
        interp_down = INTERP[args.downsample]
        target = args.output_size if args.output_size is not None else args.size
        small = cv2.resize(img_rgb, (target, target), interpolation=interp_down)
        if args.output_size is not None:
            out_rgb = small
        else:
            interp_up = INTERP[args.upsample]
            out_rgb = cv2.resize(small, (w, h), interpolation=interp_up)

    buf = io.BytesIO()
    Image.fromarray(out_rgb).save(buf, format="JPEG", quality=args.jpeg_quality)
    return buf.getvalue()


def build_output_path(src: Path, args) -> Path:
    if args.output_size is not None:
        if args.method == 'esrgan':
            tag = f"_noup_{args.output_size}_esrgan_{MODE_SUFFIX[args.mode]}"
        else:
            tag = f"_noup_{args.output_size}_{args.downsample}_{MODE_SUFFIX[args.mode]}"
    else:
        tag = f"_{args.size}_{args.downsample}_{args.upsample}_{MODE_SUFFIX[args.mode]}"
    return src.with_name(src.stem + tag + ".bin")


def process_bin(src: Path, dst: Path, args):
    bins, issame_list = load_bin(src)
    n_pairs = len(issame_list)

    degrade_first  = args.mode in ("first", "both")
    degrade_second = args.mode in ("second", "both")

    # Reproducible RNG seeded with src filename + global seed so different
    # .bin files get distinct (but reproducible) ESRGAN samples.
    rng = np.random.default_rng(args.seed + (hash(src.name) & 0xFFFFFFFF))

    new_bins = list(bins)
    for i in tqdm(range(n_pairs), desc=f"{src.name} → {dst.name}", leave=False):
        if degrade_first:
            new_bins[i * 2] = degrade(bins[i * 2], args, rng=rng)
        if degrade_second:
            new_bins[i * 2 + 1] = degrade(bins[i * 2 + 1], args, rng=rng)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump((new_bins, issame_list), f, protocol=4)

    print(f"  {src.name} -> {dst.name}  ({n_pairs} pairs)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LR-degraded .bin verification files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--bin",     type=Path, help="Single input .bin file")
    src_group.add_argument("--bin-dir", type=Path, help="Directory of input .bin files")

    parser.add_argument("--output",    type=Path,  default=None,
                        help="Output path (only valid with --bin). "
                             "Default: auto-named sibling of the source file.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (only valid with --bin-dir). "
                             "Default: same directory as the source files.")
    parser.add_argument("--method",    type=str,   default="downsample",
                        choices=["downsample", "esrgan"],
                        help="Degradation method. 'downsample' = simple resize "
                             "(first-order ESRGAN-style when --downsample cubic). "
                             "'esrgan' = Real-ESRGAN second-order stochastic pipeline "
                             "(requires --output-size).")
    parser.add_argument("--size",      type=int,   default=None, choices=[56, 28, 14, 7],
                        help="Target downsample resolution (square). The image is "
                             "upsampled back to its original H×W. Mutually exclusive "
                             "with --output-size.")
    parser.add_argument("--output-size", type=int, default=None,
                        help="Final stored resolution (no upsample-back). Produces "
                             "the noup_{size}_* output naming. Required for "
                             "--method esrgan. Mutually exclusive with --size.")
    parser.add_argument("--downsample", type=str,  default="area",
                        choices=["area", "cubic", "linear"],
                        help="Interpolation method for downsampling "
                             "(ignored for --method esrgan)")
    parser.add_argument("--upsample",  type=str,   default="area",
                        choices=["area", "cubic", "linear"],
                        help="Interpolation method for upsampling "
                             "(only used in upsample-back mode)")
    parser.add_argument("--mode",      type=str,   default="both",
                        choices=["first", "second", "both"],
                        help="Which image in each pair to degrade: "
                             "'first' (LR|HR), 'second' (HR|LR), 'both' (LR|LR)")
    parser.add_argument("--bins",      nargs="*",  default=None,
                        help="With --bin-dir: process only these stems (no extension). "
                             "Default: all .bin files.")
    parser.add_argument("--jpeg-quality", type=int, default=100,
                        help="JPEG quality for re-encoded images (default: 100)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed for stochastic methods (default: 42)")

    args = parser.parse_args()

    # Validate size flags.
    if (args.size is None) == (args.output_size is None):
        parser.error("Specify exactly one of --size or --output-size")
    if args.method == 'esrgan' and args.output_size is None:
        parser.error("--method esrgan requires --output-size")

    if args.output_size is not None:
        if args.method == 'esrgan':
            print(f"method=esrgan  output_size={args.output_size}  mode={args.mode}")
        else:
            print(f"noup  output_size={args.output_size}  down={args.downsample}  "
                  f"mode={args.mode}")
    else:
        print(f"size={args.size}  down={args.downsample}  up={args.upsample}  "
              f"mode={args.mode}")

    if args.bin:
        if not args.bin.exists():
            raise FileNotFoundError(f"Not found: {args.bin}")
        dst = args.output or build_output_path(args.bin, args)
        process_bin(args.bin, dst, args)

    else:  # --bin-dir
        bin_files = sorted(args.bin_dir.glob("*.bin"))
        if args.bins:
            bin_files = [b for b in bin_files if b.stem in args.bins]
        if not bin_files:
            print(f"No matching .bin files in {args.bin_dir}")
            return

        out_dir = args.output_dir or args.bin_dir
        print(f"Processing {len(bin_files)} file(s) from {args.bin_dir} -> {out_dir}\n")

        for src in bin_files:
            dst = out_dir / build_output_path(src, args).name
            process_bin(src, dst, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
