# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Samplers and multi-loader utilities for cache-friendly distributed training.

``ChunkedDistributedSampler`` yields indices in contiguous chunks so that
data backed by chunked storage (e.g. zarr) benefits from sequential I/O.

``RoundRobinLoader`` interleaves multiple DataLoaders — typically one per
worker, each with its own ``ChunkedDistributedSampler`` — to provide an
iterable-style interface over map-style datasets with per-worker chunk
affinity.
"""

import random

import torch
import torch.distributed
import torch.utils.data


# ---------------------------------------------------------------------------
# Chunked distributed sampler
# ---------------------------------------------------------------------------


class ChunkedDistributedSampler(torch.utils.data.Sampler):
    """A distributed sampler that yields indices in contiguous chunks.

    Within each chunk, indices are sequential (optionally shuffled within the
    chunk).  Chunks themselves can be shuffled across epochs.  This pattern is
    critical when the underlying dataset caches data at chunk granularity, as
    it ensures sequential access within each cache window.

    The sampler is infinite: after exhausting all chunks it advances the epoch
    and re-shuffles.

    Args:
        dataset: Map-style dataset.
        chunk_size: Number of contiguous indices per chunk.
        rank: This worker's global rank.
        num_replicas: Total number of workers (data-parallel * per-GPU workers).
        shuffle: Whether to shuffle the order of chunks.
        shuffle_within_chunk: Whether to shuffle indices within each chunk.
        drop_last: Whether to drop incomplete trailing chunks.
        seed: Random seed (broadcast from rank 0 when distributed).
        sampler_fn: Optional custom sampler over chunk indices.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        chunk_size: int = 1,
        rank=0,
        num_replicas=1,
        shuffle=False,
        shuffle_within_chunk=False,
        drop_last=True,
        seed=42,
        sampler_fn=None,
    ):
        super().__init__()
        self.n = len(dataset)
        nchunks = self.n // chunk_size
        chunks = list(range(nchunks))

        if torch.distributed.is_initialized():
            seed = torch.tensor(seed).cuda()
            torch.distributed.broadcast(seed, src=0)
            seed = seed.item()

        self._chunk_sampler = (
            sampler_fn(chunks)
            if sampler_fn is not None
            else torch.utils.data.DistributedSampler(
                chunks,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        )
        self.chunk_size = chunk_size
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed = seed
        self.rank = rank
        self.epoch = 0
        self.index_within_chunk = 0
        self._chunk_iter = iter(self._chunk_sampler)
        self._current_chunk_indices = None

        if self.shuffle_within_chunk:
            self.rng = random.Random(seed + rank)

    def set_epoch(self, epoch):
        try:
            self._chunk_sampler.set_epoch(epoch)
        except AttributeError:
            pass
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        if self.index_within_chunk == 0:
            try:
                self.active_chunk = next(self._chunk_iter)
            except StopIteration:
                self.set_epoch(self.epoch + 1)
                self._chunk_iter = iter(self._chunk_sampler)
                raise StopIteration()

            chunk_start = self.active_chunk * self.chunk_size
            self._current_chunk_indices = list(
                range(chunk_start, chunk_start + self.chunk_size)
            )

            if self.shuffle_within_chunk:
                self.rng.shuffle(self._current_chunk_indices)

        i = self._current_chunk_indices[self.index_within_chunk]
        self.index_within_chunk = (self.index_within_chunk + 1) % self.chunk_size
        return i


# ---------------------------------------------------------------------------
# Round-robin loader
# ---------------------------------------------------------------------------


class RoundRobinLoader(torch.utils.data.IterableDataset):
    """Round-robin interleaving of multiple map-style DataLoaders.

    This converts map-style datasets to iterable-style by cycling through
    the given DataLoaders in round-robin order, removing exhausted ones
    until all are done.

    Typical usage: create one ``DataLoader`` per worker, each backed by a
    ``ChunkedDistributedSampler`` with a unique rank, then wrap them with
    ``RoundRobinLoader``.

    Args:
        dataloaders: List of DataLoader instances to interleave.
    """

    def __init__(self, dataloaders: list[torch.utils.data.DataLoader]):
        super().__init__()
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        active_indices = list(range(len(self.dataloaders)))

        while active_indices:
            for idx in list(active_indices):
                try:
                    yield next(iterators[idx])
                except StopIteration:
                    active_indices.remove(idx)
