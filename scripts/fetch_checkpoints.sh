#!/bin/bash
#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause
# Script: fetch_checkpoints.sh
#
# Download the pretrained EdgeFace-Base weights from Hugging Face into
# checkpoints/, where the configs expect them:
#
#   bash scripts/fetch_checkpoints.sh
#
# The weights are NOT part of this repository. They are distributed at
# https://huggingface.co/Idiap/EdgeFace-Base under CC-BY-NC-SA-4.0, which is a
# different license from this repository's code (BSD-3-Clause).
#
# Only EdgeFace-Base is needed here (the frozen backbone of the SR and PDT
# pipelines and the "base, direct" baseline). Every other EdgeFace variant,
# including the quantized ones, is available from the official release:
#   https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface
#   https://github.com/otroshi/edgeface

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT/checkpoints"
BASE_URL="https://huggingface.co/Idiap/EdgeFace-Base/resolve/main"

mkdir -p "$DEST_DIR"

fetch() {
    local name="$1" md5_expected got
    local dest="$DEST_DIR/$name"

    # The published checksum file has the form:
    #   Filename: edgeface_base.pt
    #   MD5 Hash: <hex>
    md5_expected="$(curl -sfL "$BASE_URL/${name%.pt}_checksum.txt" \
                    | sed -n 's/^MD5 Hash: *//p')"

    if [ -f "$dest" ] && [ -n "$md5_expected" ] \
       && echo "$md5_expected  $dest" | md5sum -c --status 2>/dev/null; then
        echo "== $name already present and verified"
        return 0
    fi

    echo "== downloading $name"
    curl -fL --progress-bar "$BASE_URL/$name" -o "$dest"

    if [ -n "$md5_expected" ]; then
        got="$(md5sum "$dest" | cut -d' ' -f1)"
        if [ "$got" != "$md5_expected" ]; then
            echo "ERROR: checksum mismatch for $name" >&2
            echo "       expected $md5_expected, got $got" >&2
            exit 1
        fi
        echo "== checksum OK ($got)"
    else
        echo "WARNING: no checksum published for $name, skipping verification" >&2
    fi
}

fetch edgeface_base.pt

echo
echo "Done. The frozen-backbone configs now resolve:"
echo "  checkpoints/edgeface_base.pt"
