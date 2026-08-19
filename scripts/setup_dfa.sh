#!/bin/bash
#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: setup_dfa.sh
#
# Fetch the CVLface Differentiable Face Aligner (DFA) used for the TinyFace
# "DFA" alignment pipeline.
#
# The aligners are third-party models hosted on the Hugging Face Hub and are not
# redistributed with this repository. This script clones them into
# face_alignment/dfa/ at the exact revisions used for the paper. Running them
# needs the packages in requirements_eval.txt (notably omegaconf).
#
# License: the model card declares no license of its own and defers to the
# license of its training dataset, WIDER FACE (CC BY-NC-ND 4.0, non-commercial).
# In this repository the aligner is used only for evaluation-time alignment of
# TinyFace test images.
#
# Usage:
#   bash scripts/setup_dfa.sh              # resnet50 only (what the paper uses)
#   bash scripts/setup_dfa.sh --all        # resnet50 and mobilenet
#
# Requires git-lfs (the model weights are LFS objects).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DFA_DIR="$ROOT/face_alignment/dfa"

# Revisions pinned to what was used for the reported TinyFace numbers.
RESNET50_URL="https://huggingface.co/minchul/cvlface_DFA_resnet50"
RESNET50_REV="ccbffdb3d3894ae9eda3571814234de05a523d03"
MOBILENET_URL="https://huggingface.co/minchul/cvlface_DFA_mobilenet"
MOBILENET_REV="8317e6dda53d91e7074979923144c2cc08906a33"

WANT_MOBILENET=0
[ "${1:-}" = "--all" ] && WANT_MOBILENET=1

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
    echo "ERROR: git-lfs is required to fetch the aligner weights." >&2
    echo "       See https://git-lfs.com for installation instructions." >&2
    exit 1
fi

fetch_aligner() {
    local name="$1" url="$2" rev="$3"
    local dest="$DFA_DIR/$name"

    if [ -d "$dest/.git" ]; then
        echo "== $name already present at $dest, skipping clone"
    else
        echo "== cloning $name"
        mkdir -p "$DFA_DIR"
        git clone "$url" "$dest"
    fi

    echo "== checking out $rev"
    git -C "$dest" checkout --quiet "$rev"
}

fetch_aligner cvlface_DFA_resnet50 "$RESNET50_URL" "$RESNET50_REV"

if [ "$WANT_MOBILENET" = "1" ]; then
    fetch_aligner cvlface_DFA_mobilenet "$MOBILENET_URL" "$MOBILENET_REV"
fi

echo
echo "Done. Align a TinyFace split with:"
echo "  sbatch slurm/align_tinyface.run Probe dfa-resnet50 112"
