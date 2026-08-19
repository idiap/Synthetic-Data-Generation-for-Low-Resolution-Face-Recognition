#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: save_bin_samples.py
#
"""
Save sample image pairs from all .bin verification files.

Each .bin stores (bins, issame_list) where bins is a flat list of raw JPEG
bytes: index 2i and 2i+1 form pair i, labelled by issame_list[i].

Usage:
    python scripts/save_bin_samples.py
    python scripts/save_bin_samples.py --bin-dir data/webface4m --n-pairs 8 --output-dir samples/bin_samples
"""
import argparse
import io
import pickle
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = PROJECT_ROOT / "data" / "webface4m"
DEFAULT_OUT_DIR = PROJECT_ROOT / "samples" / "bin_samples"


def load_bin(bin_path: Path):
    """Load a .bin verification file, returning (bins, issame_list)."""
    try:
        with open(bin_path, "rb") as f:
            bins, issame_list = pickle.load(f)
    except UnicodeDecodeError:
        with open(bin_path, "rb") as f:
            bins, issame_list = pickle.load(f, encoding="bytes")
    return bins, issame_list


def decode_image(raw_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def make_pair_strip(img1: Image.Image, img2: Image.Image, label: str) -> Image.Image:
    """
    Combine two images side-by-side with a label banner below.
    Resizes both to the same height if needed.
    """
    h = max(img1.height, img2.height)
    img1 = img1.resize((img1.width * h // img1.height, h))
    img2 = img2.resize((img2.width * h // img2.height, h))

    banner_h = 20
    total_w  = img1.width + 4 + img2.width   # 4px gap
    strip    = Image.new("RGB", (total_w, h + banner_h), (40, 40, 40))
    strip.paste(img1, (0, 0))
    strip.paste(img2, (img1.width + 4, 0))

    draw  = ImageDraw.Draw(strip)
    color = (80, 200, 80) if label == "same" else (220, 80, 80)
    draw.rectangle([0, h, total_w, h + banner_h], fill=color)
    draw.text((4, h + 3), label, fill=(255, 255, 255))

    return strip


def save_samples(bin_path: Path, output_dir: Path, n_pairs: int, seed: int):
    print(f"  Loading {bin_path.name} ...", end=" ", flush=True)
    bins, issame_list = load_bin(bin_path)
    n_total = len(issame_list)
    print(f"{n_total} pairs")

    rng         = random.Random(seed)
    pair_indices = rng.sample(range(n_total), min(n_pairs, n_total))

    out = output_dir / bin_path.stem
    out.mkdir(parents=True, exist_ok=True)

    for i, pair_idx in enumerate(pair_indices):
        img1 = decode_image(bins[pair_idx * 2])
        img2 = decode_image(bins[pair_idx * 2 + 1])
        label = "same" if issame_list[pair_idx] else "different"

        # Save individual images
        img1.save(out / f"pair{i:02d}_idx{pair_idx}_img1.jpg")
        img2.save(out / f"pair{i:02d}_idx{pair_idx}_img2.jpg")

        # Save combined strip
        strip = make_pair_strip(img1, img2, label)
        strip.save(out / f"pair{i:02d}_idx{pair_idx}_strip_{label}.jpg")

    print(f"    -> saved {len(pair_indices)} pairs to {out}")


def main():
    parser = argparse.ArgumentParser(description="Save sample pairs from .bin verification files")
    parser.add_argument("--bin-dir",    type=Path, default=DEFAULT_BIN_DIR,
                        help="Directory containing .bin files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Root output directory")
    parser.add_argument("--n-pairs",    type=int,  default=5,
                        help="Number of pairs to sample per .bin file")
    parser.add_argument("--seed",       type=int,  default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--bins",       nargs="*", default=None,
                        help="Specific .bin names to process (without extension). "
                             "Default: all .bin files in --bin-dir")
    args = parser.parse_args()

    bin_files = sorted(args.bin_dir.glob("*.bin"))
    if not bin_files:
        print(f"No .bin files found in {args.bin_dir}")
        return

    if args.bins:
        bin_files = [b for b in bin_files if b.stem in args.bins]
        if not bin_files:
            print(f"None of the requested bins found: {args.bins}")
            return

    print(f"Found {len(bin_files)} .bin file(s) in {args.bin_dir}")
    print(f"Saving {args.n_pairs} sample pairs each -> {args.output_dir}\n")

    for bin_path in bin_files:
        save_samples(bin_path, args.output_dir, args.n_pairs, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
