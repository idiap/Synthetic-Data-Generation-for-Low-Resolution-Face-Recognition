#
# SPDX-FileCopyrightText: Copyright (c) 2022 Jiankang Deng and Jia Guo
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Script: dataset_pdt.py
# Modified from InsightFace (https://github.com/deepinsight/insightface, MIT);
# see LICENSES/MIT.txt. Changes: PK-sampled paired LR/HR loading for contrastive PDT training.
#
"""
PDT training dataloaders — MXNet-free RecordIO backend + PK batch sampler.

RecFaceDataset
    Reads MXNet RecordIO files directly via RecordIODataManager (no MXNet
    dependency).  Labels are scanned once at first use and cached as a
    labels.npy file in the dataset directory.  Each DataLoader worker opens
    its own file handle, avoiding cross-process file-handle sharing.

PKBatchSampler
    Samples P identities × K images per batch (the standard PK / M-per-class
    schedule for metric learning).  Distributed-aware: at each global step,
    all P*world_size identities are drawn from the same shuffled sequence, so
    rank r gets a contiguous slice of P identities.

    Using the SAME seed for both the LR and HR loaders guarantees that both
    loaders visit the same set of identities at every step → positive
    cross-modal pairs are guaranteed in every batch.

    Image selection within each identity is drawn from the shared RNG *after*
    the identity shuffle, so the two loaders independently pick different images
    of the same identity → diverse positive pairs.

get_pdt_dataloader
    Factory that wires RecFaceDataset + PKBatchSampler + DataLoaderX together.
"""

import io
import os
import struct
import threading
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from PIL import ImageFile
# Allow PIL to decode truncated JPEG records (e.g. partially-written .rec files)
# instead of raising OSError, which would crash DataLoader workers.
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from recordio import RecordIODataManager
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class _RandomDownscale(torch.nn.Module):
    """Simulate low-resolution degradation: shrink to a random size then restore.

    Mirrors the DALI dali_aug random-resize step:
        fn.resize(img, random_size)  →  fn.resize(img, 112)
    where random_size ~ Uniform(int(112*0.5), int(112*0.8)) = [56, 89].
    """
    def __init__(self, min_size: int = 56, max_size: int = 89, p: float = 0.1):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.p = p

    def forward(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            size = torch.randint(self.min_size, self.max_size + 1, (1,)).item()
            img = transforms.functional.resize(
                img, [size, size],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True)
            img = transforms.functional.resize(
                img, [112, 112],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True)
        return img


def _build_transform(augment: bool = False) -> transforms.Compose:
    """Build the face-recognition preprocessing pipeline.

    Base pipeline (augment=False) matches MXFaceDataset and the DALI path:
        RandomHorizontalFlip → ToTensor → Normalize(0.5, 0.5)  →  [-1, 1]

    Optional augmentations (augment=True) replicate the DALI dali_aug pipeline:
        • _RandomDownscale   p=0.10  — random downscale+upscale (56-89 → 112)
        • GaussianBlur       p=0.20  — kernel randomly 3 or 5
        • ColorJitter        p=0.20  — hue ±10°, saturation ×1.0–1.2
        • RandomGrayscale    p=0.10  — saturation → 0 (RGB image kept 3-channel)

    The augment flag is disabled by default to preserve consistency with
    previous HR trainings until explicit testing on PDT is done.
    """
    pipeline: list = [transforms.RandomHorizontalFlip()]

    if augment:
        pipeline += [
            _RandomDownscale(min_size=56, max_size=89, p=0.1),
            # kernel_size=[3, 5]: torchvision picks a random odd value in [3, 5]
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=[3, 5])], p=0.2),
            # hue: DALI range [0°, 20°] → torchvision fraction ±(10/360)
            # saturation: DALI multiplier [1.0, 1.2] → torchvision (min, max) tuple
            transforms.RandomApply(
                [transforms.ColorJitter(hue=10/360, saturation=(1.0, 1.2))], p=0.2),
            transforms.RandomGrayscale(p=0.1),
        ]

    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(pipeline)


_DEFAULT_TRANSFORM = _build_transform(augment=False)

# RecordIO record-header constants (mirrors RecordIODataManager)
_IR_FORMAT = "IfQQ"
_IR_SIZE = struct.calcsize(_IR_FORMAT)


# ---------------------------------------------------------------------------
# Prefetch / background-thread DataLoader (from dataset.py, copied to avoid
# importing dataset.py which has a top-level `import mxnet as mx`)
# ---------------------------------------------------------------------------
import queue as Queue


class _BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super().__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def __iter__(self):
        return self


class _DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = _BackgroundGenerator(self.iter, self.local_rank)
        self._preload()
        return self

    def _preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(
                    device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RecFaceDataset(Dataset):
    """
    PyTorch Dataset backed by MXNet RecordIO files.  No MXNet required.

    At init the full label array is scanned (using only record headers, not
    image data) and cached as labels.npy.  Subsequent runs load the cache
    directly — for WebFace4M (~4.2 M records) the first scan takes about
    1–2 min; cached loads are instant.

    Worker safety: each DataLoader worker (separate process) opens its own
    RecordIODataManager file handle on the first __getitem__ call, so there
    are no cross-process race conditions on the shared .rec file.
    """

    def __init__(self, root_dir: str, transform=None, augment: bool = False,
                 labels_cache: Optional[str] = None):
        self.root_dir = Path(root_dir)
        self.rec_path = str(self.root_dir / "train.rec")
        self.idx_path = str(self.root_dir / "train.idx")
        # Explicit transform takes priority; augment flag is used only when
        # no transform is provided.
        self.transform = transform if transform is not None else _build_transform(augment)
        self._local = threading.local()  # per-worker file handles

        # ---- read metadata from header record (position 0) ----
        dm = RecordIODataManager(self.idx_path, self.rec_path)
        meta_bytes = dm.get_raw_bytes(0)
        meta_header, _ = RecordIODataManager._unpack_recordio(meta_bytes)

        if meta_header.flag > 0:
            # label[0] = num_imgs, label[1] = num_classes
            num_imgs = int(meta_header.label[0])
            # positions 1 … num_imgs-1 are the image records
            self.imgidx = np.arange(1, num_imgs, dtype=np.int64)
        else:
            self.imgidx = np.arange(1, len(dm), dtype=np.int64)
        dm.close()

        # ---- load or build label cache (rank-0 only to avoid concurrent I/O) ----
        # All N ranks × 2 loaders used to scan the full .rec file simultaneously
        # at startup, exhausting the OS page cache and RAM.
        # labels_cache lets the HR loader reuse the LR loader's cache — since
        # both datasets share the same record order and identity labels, only
        # one scan is ever needed.
        cache_path = Path(labels_cache) if labels_cache else self.root_dir / "labels.npy"
        rank = int(os.environ.get("RANK", 0))
        if not cache_path.exists():
            if rank == 0:
                print(f"[RecFaceDataset] Building label cache for {self.root_dir} …")
                labels = self._build_labels()
                np.save(str(cache_path), labels)
                print(f"[RecFaceDataset] Saved label cache: {cache_path}")
            # All non-zero ranks wait here until rank 0 has written the cache.
            from torch import distributed as _dist
            if _dist.is_initialized():
                _dist.barrier()
        self.labels = np.load(str(cache_path))
        if rank == 0:
            print(f"[RecFaceDataset] Loaded label cache: {cache_path}")

        # ---- class → sample-indices mapping ----
        class_to_indices: Dict[int, list] = {}
        for sample_idx, label in enumerate(self.labels):
            lbl = int(label)
            if lbl not in class_to_indices:
                class_to_indices[lbl] = []
            class_to_indices[lbl].append(sample_idx)
        self.class_to_indices: Dict[int, np.ndarray] = {
            cls: np.array(idxs, dtype=np.int64)
            for cls, idxs in class_to_indices.items()
        }
        self.classes = sorted(self.class_to_indices.keys())

    # ------------------------------------------------------------------
    # Label scanning (reads only the 24-byte IR header per record,
    # not the image payload — roughly 20× faster than full read)
    # ------------------------------------------------------------------
    def _build_labels(self) -> np.ndarray:
        labels = []
        # Load key→offset mapping directly (avoid opening full RecordIODataManager)
        idx_map: Dict[int, int] = {}
        keys_ordered = []
        with open(self.idx_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    key = int(parts[0])
                    idx_map[key] = int(parts[1])
                    keys_ordered.append(key)

        # Skip key 0 (metadata record)
        image_keys = keys_ordered[1:]

        with open(self.rec_path, "rb") as rec_file:
            for key in tqdm(image_keys, desc="Scanning record labels"):
                offset = idx_map[key]
                # RecordIO frame: 4 B magic | 4 B cflag_length | <length> B payload
                # payload starts with the IR header (flag, label, id, id2)
                rec_file.seek(offset + 8)  # skip magic + cflag_length
                hdr = struct.unpack(_IR_FORMAT, rec_file.read(_IR_SIZE))
                # hdr[1] is the float32 class label
                labels.append(int(hdr[1]))
        return np.array(labels, dtype=np.int32)

    # ------------------------------------------------------------------
    # Per-worker file handle
    # ------------------------------------------------------------------
    def _get_dm(self) -> RecordIODataManager:
        if not hasattr(self._local, "dm"):
            self._local.dm = RecordIODataManager(self.idx_path, self.rec_path)
        return self._local.dm

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        dm = self._get_dm()
        pos = int(self.imgidx[index])
        data = dm.get_raw_bytes(pos)
        _, img_bytes = RecordIODataManager._unpack_recordio(data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return len(self.imgidx)


# ---------------------------------------------------------------------------
# PK batch sampler
# ---------------------------------------------------------------------------

class PKBatchSampler(Sampler):
    """
    Samples P identities × K images per batch in a distributed-safe manner.

    Identity shuffle order
    ----------------------
    All ranks share the same RNG seed, so they all compute the same shuffled
    identity sequence.  At each global step s, rank r takes the P identities
    at positions [s*P*W + r*P : s*P*W + (r+1)*P] of that sequence.  This
    ensures every rank covers different (non-overlapping) identities and that
    the total number of batches is identical across all ranks (no deadlocks).

    Cross-modal positive pairs
    --------------------------
    When both LR and HR loaders are built with the SAME seed, they generate
    the same identity sequence → same identities per batch per step → at
    least P*K² positive cross-modal pairs guaranteed per batch.

    Image selection within each identity uses the RNG state *after* the
    identity shuffle, so the two loaders will independently pick different
    images of the same person (diverse positives) because the RNG will have
    advanced differently between iterations — unless both are iterated in
    strict lockstep, in which case samples are identical but still valid
    positives.

    Args:
        class_to_indices : {class_id: np.ndarray of 0-based sample indices}
        p                : identities per batch
        k                : images per identity (with replacement if class has fewer)
        rank, world_size : for distributed training
        seed             : base seed; combined with epoch via set_epoch()
    """

    def __init__(
        self,
        class_to_indices: Dict[int, np.ndarray],
        p: int,
        k: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.p = p
        self.k = k
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.class_to_indices = class_to_indices

        all_classes = np.array(sorted(class_to_indices.keys()), dtype=np.int64)
        # Trim to a multiple of (p * world_size) so every rank gets the same
        # number of batches — avoids distributed deadlocks.
        n_usable = (len(all_classes) // (p * world_size)) * (p * world_size)
        self._all_classes = all_classes[:n_usable]
        self.num_batches = n_usable // (p * world_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled = rng.permutation(self._all_classes)  # shared order for all ranks

        for step in range(self.num_batches):
            base = step * self.p * self.world_size + self.rank * self.p
            batch_classes = shuffled[base : base + self.p]

            batch_indices = []
            for cls in batch_classes:
                cls_indices = self.class_to_indices[int(cls)]
                if len(cls_indices) >= self.k:
                    chosen = rng.choice(cls_indices, self.k, replace=False)
                else:
                    chosen = rng.choice(cls_indices, self.k, replace=True)
                batch_indices.extend(chosen.tolist())
            yield batch_indices


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_pdt_dataloader(
    root_dir: str,
    local_rank: int,
    p: int,
    k: int,
    seed: int = 2048,
    num_workers: int = 4,
    transform=None,
    augment: bool = False,
    labels_cache: Optional[str] = None,
) -> _DataLoaderX:
    """
    Build a prefetching DataLoader backed by RecFaceDataset + PKBatchSampler.

    Both the LR and HR loaders for PDT training should be created with the
    SAME (p, k, seed) so that every batch contains the same set of identities
    in both modalities.

    labels_cache: optional path to a pre-built labels.npy file.  Since LR and
    HR datasets share the same record order and identity labels, pass the path
    of the LR loader's cache to the HR loader to skip the second scan entirely:

        lr_loader = get_pdt_dataloader(cfg.rec,    ..., labels_cache=None)
        hr_loader = get_pdt_dataloader(cfg.rec_hr, ...,
                        labels_cache=str(Path(cfg.rec) / "labels.npy"))

    Returns a _DataLoaderX instance whose .batch_sampler is a PKBatchSampler;
    call loader.batch_sampler.set_epoch(epoch) at the start of every epoch.
    """
    rank, world_size = get_dist_info()

    dataset = RecFaceDataset(root_dir, transform=transform, augment=augment,
                             labels_cache=labels_cache)

    sampler = PKBatchSampler(
        class_to_indices=dataset.class_to_indices,
        p=p,
        k=k,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    loader = _DataLoaderX(
        local_rank=local_rank,
        dataset=dataset,
        batch_sampler=sampler,   # overrides batch_size / sampler / drop_last
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=init_fn,
    )
    return loader
