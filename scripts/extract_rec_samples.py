#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: extract_rec_samples.py
#
"""
Extract the first N images from a .rec file and save them as JPEGs
for visual inspection.
"""
import argparse
import os
import numbers

import mxnet as mx
import numpy as np
import cv2


def extract_samples(rec_dir, output_dir, n=6):
    path_imgrec = os.path.join(rec_dir, "train.rec")
    path_imgidx = os.path.join(rec_dir, "train.idx")

    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")

    # Read metadata record at index 0
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)

    if header.flag > 0:
        total_images = int(header.label[0])
        print(f"Metadata record found: {total_images} images, "
              f"num_classes={int(header.label[1])}")
        start_idx = 1
    else:
        print("No metadata record at index 0; treating index 0 as first image.")
        start_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        idx = start_idx + i
        s = imgrec.read_idx(idx)
        if s is None:
            print(f"Index {idx}: no record found.")
            continue

        header, img_bytes = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]

        # Decode image via MXNet (returns HWC RGB)
        img_mx = mx.image.imdecode(img_bytes)
        img_np = img_mx.asnumpy()  # RGB uint8

        # Convert to BGR for cv2.imwrite
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_dir, f"idx{idx:06d}_label{int(label)}.jpg")
        cv2.imwrite(out_path, img_bgr)
        print(f"  rec_idx={idx}  label={int(label)}  "
              f"shape={img_np.shape}  "
              f"min={img_np.min()}  max={img_np.max()}  "
              f"saved → {out_path}")

    imgrec.close()
    print(f"\nDone. {n} images saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract sample images from an MXNet .rec file.")
    parser.add_argument("rec_dir",
                        help="Directory containing train.rec and train.idx")
    parser.add_argument("--output", default="rec_samples",
                        help="Output directory for extracted images (default: rec_samples)")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of images to extract (default: 6)")
    args = parser.parse_args()
    extract_samples(args.rec_dir, args.output, args.n)
