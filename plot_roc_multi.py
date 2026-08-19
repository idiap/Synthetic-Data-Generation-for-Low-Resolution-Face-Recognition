#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: plot_roc_multi.py
#
"""
Load several scores_labels.npz files (same format as eval_video_verification.write_results)
and plot a single ROC figure with one curve per file. The legend label is the file stem.

Example:
  python plot_roc_multi.py --scores-dir /path/to/folder -o /path/to/roc_multi.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

LOG = logging.getLogger(__name__)


def _load_scores_labels(path: Path):
    data = np.load(path, allow_pickle=False)
    if "scores" not in data or "labels" not in data:
        raise KeyError(
            f"{path}: expected keys 'scores' and 'labels' (eval_video_verification .npz format)."
        )
    return np.asarray(data["scores"], dtype=np.float32), np.asarray(
        data["labels"], dtype=np.int32
    )


def _roc_for_file(path: Path):
    scores, labels = _load_scores_labels(path)
    if np.unique(labels).shape[0] < 2:
        return None, None, None
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, float(roc_auc)


def plot_multi_roc(npz_paths, output_path, title: str = "Video Verification ROC"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plotted = 0
    for p in npz_paths:
        name = p.stem
        fpr, tpr, roc_auc = _roc_for_file(p)
        if fpr is None:
            LOG.warning("Skipping %s: need both positive and negative labels for ROC.", p)
            continue
        plt.plot(fpr, tpr, lw=1, label=f"{name} (AUC={roc_auc:.4f})")
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No valid ROC curves to plot (all files skipped or empty).")

    plt.xscale("log")
    plt.xlim([1e-6, 1e-1])
    plt.ylim([0.0, 1.0])
    plt.grid(linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOG.info("Saved ROC plot to %s (%d series).", output_path, plotted)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot multiple verification ROC curves from scores_labels .npz files."
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        required=True,
        help="Directory containing .npz files (scores + labels per model).",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern under --scores-dir (default: all .npz).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <scores-dir>/roc_multi.pdf).",
    )
    parser.add_argument(
        "--title",
        default="Video Verification ROC",
        help="Plot title.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    d = args.scores_dir.resolve()
    if not d.is_dir():
        LOG.error("Not a directory: %s", d)
        return 1

    files = sorted(d.glob(args.pattern))
    if not files:
        LOG.error("No files matching %s in %s", args.pattern, d)
        return 1

    out = args.output if args.output is not None else d / "roc_multi.pdf"
    try:
        plot_multi_roc(files, out, title=args.title)
    except Exception as e:
        LOG.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
