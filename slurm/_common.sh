#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: _common.sh
#
# Shared prelude for the SLURM launchers in slurm/. Not executable on its own;
# every launcher sources it as its first statement:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# It loads slurm/env.sh (falling back to slurm/env.sh.example), activates the
# requested conda environment, moves to the repository root, and exposes the
# `stage_rec` and `master_port` helpers.

set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$_SLURM_DIR/env.sh" ]; then
    # shellcheck source=/dev/null
    source "$_SLURM_DIR/env.sh"
else
    echo "NOTE: slurm/env.sh not found, using defaults from env.sh.example." >&2
    echo "      Run 'cp slurm/env.sh.example slurm/env.sh' to customise." >&2
    # shellcheck source=/dev/null
    source "$_SLURM_DIR/env.sh.example"
fi

export LRFR_DATA_ROOT TINYFACE_ROOT IJBC_ROOT

cd "$CODEBASE_PATH"
mkdir -p logs

if [ "${LOAD_CLUSTER_MODULES}" = "1" ]; then
    module --ignore_cache load cuda
    module --ignore_cache load cudnn
fi

# activate_env [train|eval]  — default: train
activate_env() {
    local which_env="${1:-train}" target
    case "$which_env" in
        eval) target="$CONDA_ENV_EVAL" ;;
        *)    target="$CONDA_ENV" ;;
    esac
    if [ ! -x "$CONDA_BIN" ]; then
        echo "ERROR: conda binary not found at '$CONDA_BIN'." >&2
        echo "       Set CONDA_BIN in slurm/env.sh." >&2
        exit 1
    fi
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$target"
}

# master_port  — a per-job torchrun port, so concurrent runs do not collide.
master_port() {
    echo $((29500 + (${SLURM_JOB_ID:-$$} % 1000)))
}

# stage_rec <dest-name> <rec-source> [bin-source]
#
# Materialise "$STAGE_DIR/<dest-name>" holding train.rec/.idx/.lst taken from
# "$LRFR_DATA_ROOT/<rec-source>" and, optionally, the verification .bin files
# from "$LRFR_DATA_ROOT/<bin-source>".
#
# The KD and PDT recipes read an LR and an HR record at once and expect them
# under specific names; this is what builds those views. Hard links are used
# when source and destination share a filesystem, so nothing is duplicated.
stage_rec() {
    local dest="$1" rec_src="$2" bin_src="${3:-}"
    local dest_dir="$STAGE_DIR/$dest"

    mkdir -p "$dest_dir"

    local src_dir="$LRFR_DATA_ROOT/$rec_src"
    if [ ! -d "$src_dir" ]; then
        echo "ERROR: dataset '$rec_src' not found under LRFR_DATA_ROOT ($LRFR_DATA_ROOT)." >&2
        echo "       Generate it first, see the Data preparation section of README.md." >&2
        exit 1
    fi

    echo "staging $rec_src -> $dest_dir"
    _link_or_copy "$src_dir"/train.* "$dest_dir/"

    if [ -n "$bin_src" ]; then
        echo "staging $bin_src/*.bin -> $dest_dir"
        _link_or_copy "$LRFR_DATA_ROOT/$bin_src"/*.bin "$dest_dir/"
    fi
}

# Hard-link when possible (same filesystem), copy otherwise. Existing files are
# left alone so a resumed job does not re-transfer the dataset.
_link_or_copy() {
    local dest_dir="${*: -1}"
    local src
    for src in "${@:1:$#-1}"; do
        [ -e "$src" ] || continue
        local target="$dest_dir/$(basename "$src")"
        [ -e "$target" ] && continue
        ln "$src" "$target" 2>/dev/null || cp "$src" "$target"
    done
}

# Once anything has been staged, the configs must read from the staging area:
# configs/paths.py resolves every dataset against LRFR_DATA_ROOT.
use_staged_data() {
    export LRFR_DATA_ROOT="$STAGE_DIR"
    echo "LRFR_DATA_ROOT is now $LRFR_DATA_ROOT"
}
