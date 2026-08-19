#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: paths.py
#
"""Filesystem roots used by the config files.

Every dataset location in ``configs/`` is expressed relative to a single root so
that the repository contains no machine-specific absolute paths. Override the
root with the ``LRFR_DATA_ROOT`` environment variable::

    export LRFR_DATA_ROOT=/path/to/your/datasets
    torchrun --nproc_per_node=4 train_v2.py configs/edgeface_s_gamma_05_lr_56_cubic_area.py

When unset it defaults to the ``data/`` directory inside the repository, which is
where ``scripts/preprocess_rec.py`` and ``scripts/merge_rec_parts.py`` write by
default.

On a cluster it is usually worth staging the ``.rec``/``.idx`` files onto
node-local storage before training and pointing ``LRFR_DATA_ROOT`` at that copy;
see ``slurm/env.sh.example``.
"""

import os

#: Root directory holding the preprocessed ``.rec``/``.idx``/``.bin`` datasets.
DATA_ROOT = os.environ.get("LRFR_DATA_ROOT", "data")


def data_path(*parts: str) -> str:
    """Join ``parts`` onto :data:`DATA_ROOT` and return a trailing-slash path.

    The training code treats ``config.rec`` as a directory, so the trailing
    separator is kept for compatibility with the original InsightFace configs.
    """
    return os.path.join(DATA_ROOT, *parts) + os.sep
