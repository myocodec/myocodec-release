"""Window-shard datasets.

Pretraining reads from pre-extracted window shards rather than streaming the corpus
HDF5 files directly. That is not an optimisation detail, it is the only version that
keeps the GPU fed: random access into the per-dataset HDF5 files (hundreds of GB, with
per-recording decompression) starves the loader, and so does a sequential read with a
shuffle buffer once the buffer drains.

`ShardDataset` therefore memory-maps each shard and reads it in large SEQUENTIAL blocks,
shuffling only *within* a block and across shard order. Do not shuffle block order: on a
shard set larger than RAM that turns every read into a random one, and throughput decays
as the page cache churns.

See `extract_shards.py` for how the shards are produced.
"""
from __future__ import annotations

import glob
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from streaming_emg_codec.config import DataConfig


class ShardDataset(torch.utils.data.IterableDataset):
    """Low-RAM block-wise streamer over float16 HDF5 window shards.

    Each shard holds one `windows` dataset of shape [n_windows, n_samples]. Windows are
    already conditioned and normalised at extraction time, so this class only reads and
    shuffles: it applies no further preprocessing.

    Infinite by design -- it cycles shards forever and the trainer stops on step count.
    """

    BLOCK = 16384                      # rows per sequential read (~0.5 GB fp32 at 10k samples)

    def __init__(self, config: DataConfig, training: bool = True):
        del training                   # shards are split at extraction time, not here
        self.shards = sorted(glob.glob(os.path.join(config.root, "*.h5")))
        if not self.shards:
            raise ValueError(f"no shards in {config.root}")

    def __iter__(self):
        import h5py

        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        rng = random.Random(int(info.seed) if info else 0)
        while True:
            order = self.shards[:]
            rng.shuffle(order)
            mine = [s for i, s in enumerate(order) if i % nw == wid] or [order[wid % len(order)]]
            for path in mine:
                try:
                    handle = h5py.File(path, "r")
                    windows = handle["windows"]
                except Exception:
                    continue
                n = windows.shape[0]
                for start in range(0, n, self.BLOCK):
                    block = np.asarray(windows[start:start + self.BLOCK], dtype=np.float32)
                    idx = list(range(block.shape[0]))
                    rng.shuffle(idx)
                    for j in idx:
                        # copy: the yielded window must not be a view into the block, or
                        # the whole block stays resident until every window is consumed
                        yield {"emg": torch.from_numpy(block[j][None, :].copy())}
                handle.close()


class SyntheticEMGDataset(Dataset):
    """Band-limited sinusoids plus noise. For smoke tests only -- not used in the paper."""

    def __init__(self, config: DataConfig, length: int = 1024):
        self.channels = config.channels or 16
        self.samples = int(config.window_seconds * config.sample_rate)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        time = torch.linspace(0, 1, self.samples)
        freqs = torch.linspace(15, 140, self.channels).unsqueeze(1)
        phase = torch.rand(self.channels, 1) * 2 * math.pi
        signal = torch.sin(2 * math.pi * freqs * time.unsqueeze(0) + phase)
        envelope = 0.5 + torch.rand(self.channels, 1)
        noise = 0.05 * torch.randn_like(signal)
        return {"emg": (signal * envelope + noise).float()}


def build_dataset(config: DataConfig, training: bool = True) -> Dataset:
    kind = config.kind.lower()
    if kind == "shards":
        return ShardDataset(config, training=training)
    if kind == "synthetic":
        return SyntheticEMGDataset(config)
    raise ValueError(f"Unknown dataset kind: {config.kind}")
