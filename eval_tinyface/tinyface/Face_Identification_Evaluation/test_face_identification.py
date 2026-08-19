#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: test_face_identification.py
# Python port of test_face_identification.m from the TinyFace identification
# protocol (https://qmul-tinyface.github.io/). The original MATLAB code and the
# .mat pair files ship with the TinyFace dataset and carry no explicit license.
#
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from compute_AP import compute_AP


def load_mat(path, key):
    """Load a variable from a MATLAB v7.3 HDF5 .mat file.

    MATLAB stores arrays in column-major order; h5py reads them transposed
    relative to MATLAB's layout, so 2-D arrays are transposed back here to
    restore the expected [n_samples x feature_dim] shape.
    """
    with h5py.File(path, "r") as f:
        data = f[key][:]
    if data.ndim == 2:
        data = data.T
    return data.squeeze()


def main(args):
    eval_dir = Path(args.eval_dir)
    feature_folder = Path(args.feature_folder) if args.feature_folder else eval_dir / "features"

    # Load gallery/probe ID pairs
    gallery_ids = load_mat(eval_dir / "gallery_match_img_ID_pairs.mat", "gallery_ids").flatten()
    probe_ids   = load_mat(eval_dir / "probe_img_ID_pairs.mat",          "probe_ids").flatten()

    # Load feature maps
    gallery_feature_map    = load_mat(feature_folder / "gallery.mat",    "gallery_feature_map")
    probe_feature_map      = load_mat(feature_folder / "probe.mat",      "probe_feature_map")
    distractor_feature_map = load_mat(feature_folder / "distractor.mat", "distractor_feature_map")

    # Build full gallery: matched gallery + distractors
    # Distractors are labelled -100; matched gallery keeps its original id
    distractor_ids      = -100 * np.ones(distractor_feature_map.shape[0], dtype=gallery_ids.dtype)
    gallery_feature_map = np.vstack([gallery_feature_map, distractor_feature_map])
    gallery_ids         = np.concatenate([gallery_ids, distractor_ids])

    # Pairwise euclidean distance matrix: [n_gallery x n_probe]
    dist = cdist(gallery_feature_map, probe_feature_map, metric="euclidean")

    ap_list  = []
    cmc_list = []

    for p in tqdm(range(probe_feature_map.shape[0]), desc="Evaluating probes"):
        probe_id   = probe_ids[p]
        good_index = np.where(gallery_ids == probe_id)[0]   # 0-based, matching compute_AP.py
        score      = dist[:, p]
        index      = np.argsort(score)                       # ascending sort (closest first)
        ap_p, cmc_p = compute_AP(good_index, index)
        ap_list.append(ap_p)
        cmc_list.append(cmc_p)

    CMC = np.mean(np.stack(cmc_list), axis=0)
    mAP = float(np.mean(ap_list))

    # Ranks are 1-based in the original MATLAB output; CMC is 0-based here
    print(
        f"mAP = {mAP:.6f}, "
        f"r1 precision = {CMC[0]:.6f}, "
        f"r5 precision = {CMC[4]:.6f}, "
        f"r10 precision = {CMC[9]:.6f}, "
        f"r20 precision = {CMC[19]:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyFace face identification evaluation (Python port of test_face_identification.m)"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=".",
        help="Directory containing gallery_match_img_ID_pairs.mat and probe_img_ID_pairs.mat "
             "(default: current directory)",
    )
    parser.add_argument(
        "--feature-folder",
        type=str,
        default=None,
        help="Directory containing gallery.mat, probe.mat, distractor.mat "
             "(default: <eval-dir>/features/)",
    )
    main(parser.parse_args())
