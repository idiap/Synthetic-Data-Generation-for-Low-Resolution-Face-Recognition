#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: generate_shuffle_index.py
#
"""
Generate a globally-shuffled index file for use with preprocess_rec.py
--shuffle-index-file.

This is needed when running SLURM array jobs for passthrough shuffling:
without a shared global permutation, each array task only shuffles its own
slice of the dataset, which does NOT produce a globally shuffled output.

Usage:
    python scripts/generate_shuffle_index.py data/webface4m \
        --output data/webface4m/shuffled_index.txt --seed 42

Then pass --shuffle-index-file data/webface4m/shuffled_index.txt to
preprocess_rec.py in your SLURM array job.
"""
import argparse
import os

import mxnet as mx
import numpy as np


def build_imgidx(rec_dir, lst_file=None):
    path_imgidx = os.path.join(rec_dir, "train.idx")
    path_imgrec = os.path.join(rec_dir, "train.rec")
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")

    if lst_file is None:
        possible_lst = os.path.join(rec_dir, "train.lst")
        if os.path.exists(possible_lst):
            lst_file = possible_lst

    if lst_file and os.path.exists(lst_file):
        print(f"Reading indices from {lst_file}...")
        imgidx_list = []
        with open(lst_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    imgidx_list.append(int(parts[0]))
        imgidx = np.array(imgidx_list, dtype=np.int64)
        print(f"  {len(imgidx)} indices")
    else:
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            n = int(header.label[0])
            print(f"Metadata record: {n} images")
            imgidx = np.arange(1, n, dtype=np.int64)
        else:
            print("No metadata, scanning all keys...")
            imgidx = np.array(sorted(imgrec.keys), dtype=np.int64)
            print(f"  {len(imgidx)} keys found")

    imgrec.close()
    return imgidx


def main():
    parser = argparse.ArgumentParser(
        description="Generate a globally-shuffled index file for preprocess_rec.py")
    parser.add_argument("rec_dir", help="Directory containing train.rec and train.idx")
    parser.add_argument("--lst", type=str, default=None,
                        help="Path to .lst file (auto-detects train.lst if not given)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: <rec_dir>/shuffled_index.txt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    imgidx = build_imgidx(args.rec_dir, args.lst)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(imgidx)

    out_path = args.output or os.path.join(args.rec_dir, "shuffled_index.txt")
    np.savetxt(out_path, imgidx, fmt="%d")
    print(f"Saved {len(imgidx)} shuffled indices → {out_path}")
    print(f"Seed: {args.seed}")
    print(f"\nTo use with a SLURM array job (e.g. 64 tasks):")
    print(f"  python scripts/preprocess_rec.py {args.rec_dir} \\")
    print(f"      --method passthrough \\")
    print(f"      --shuffle-index-file {out_path} \\")
    print(f"      --slurm-array-count 64")


if __name__ == "__main__":
    main()
