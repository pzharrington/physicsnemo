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
"""Tests for ChunkedDistributedSampler and RoundRobinLoader."""

import itertools

import torch
import torch.utils.data

from physicsnemo.experimental.datapipes.healda.samplers import (
    ChunkedDistributedSampler,
    RoundRobinLoader,
)


def test_chunked_sampler_sequential():
    """Indices within a chunk must be consecutive."""
    s = ChunkedDistributedSampler(list(range(100)), chunk_size=5)
    it = iter(s)
    visited = set()
    for chunk in range(20):
        last_i = 0
        for i in range(5):
            idx = next(it)
            if i > 0:
                assert idx - last_i == 1
            last_i = idx
            visited.add(idx)

    assert len(visited) == 100


def test_chunked_sampler_with_islice():
    """Verify iter(sampler) continues rather than resetting."""
    dataset = list(range(100))
    sampler = ChunkedDistributedSampler(dataset, chunk_size=10, drop_last=False)

    iterator = iter(sampler)
    first_10 = list(itertools.islice(iterator, 10))
    assert first_10 == list(range(10))

    # Re-calling iter should continue, not restart
    iterator2 = iter(sampler)
    next_10 = list(itertools.islice(iterator2, 10))
    assert next_10 == list(range(10, 20))
    assert first_10 != next_10


def test_shuffle_within_chunk():
    """Within-chunk shuffle randomizes order but preserves membership."""
    s = ChunkedDistributedSampler(
        list(range(100)),
        chunk_size=10,
        shuffle=False,
        shuffle_within_chunk=True,
        seed=42,
    )

    indices = list(s)
    assert sorted(indices) == list(range(100))

    first_chunk = indices[:10]
    assert sorted(first_chunk) == list(range(10))
    assert first_chunk != list(range(10))  # order should differ


def test_shuffle_epoch_changes_chunks():
    """Epoch auto-increment produces different chunk orderings."""
    s = ChunkedDistributedSampler(
        list(range(100)),
        chunk_size=10,
        shuffle=True,
        shuffle_within_chunk=True,
        seed=42,
    )

    epoch1 = list(s)
    epoch2 = list(s)

    assert sorted(epoch1) == list(range(100))
    assert sorted(epoch2) == list(range(100))
    assert sorted(epoch1[:10]) != sorted(epoch2[:10])


# ---------------------------------------------------------------------------
# RoundRobinLoader tests
# ---------------------------------------------------------------------------


def test_round_robin_loader():
    """Round-robin interleaving across three loaders."""
    loader1 = torch.utils.data.DataLoader(list(range(0, 10)), batch_size=2)
    loader2 = torch.utils.data.DataLoader(list(range(10, 15)), batch_size=2)
    loader3 = torch.utils.data.DataLoader(list(range(15, 20)), batch_size=2)

    rr = RoundRobinLoader([loader1, loader2, loader3])
    assert len(rr) == len(loader1) + len(loader2) + len(loader3)

    batches = list(rr)
    assert len(batches) == 11

    # First round
    assert torch.equal(batches[0], torch.tensor([0, 1]))
    assert torch.equal(batches[1], torch.tensor([10, 11]))
    assert torch.equal(batches[2], torch.tensor([15, 16]))


def test_round_robin_loader_uneven():
    """Uneven loader lengths — shorter ones drop out first."""
    loader1 = torch.utils.data.DataLoader(list(range(0, 20)), batch_size=2)
    loader2 = torch.utils.data.DataLoader(list(range(20, 22)), batch_size=2)

    rr = RoundRobinLoader([loader1, loader2])
    batches = list(rr)
    assert len(batches) == 11

    assert torch.equal(batches[0], torch.tensor([0, 1]))
    assert torch.equal(batches[1], torch.tensor([20, 21]))
    assert torch.equal(batches[2], torch.tensor([2, 3]))


def test_round_robin_loader_empty():
    rr = RoundRobinLoader([])
    assert list(rr) == []


def test_round_robin_loader_single():
    loader = torch.utils.data.DataLoader(list(range(10)), batch_size=3)
    rr = RoundRobinLoader([loader])
    expected = list(torch.utils.data.DataLoader(list(range(10)), batch_size=3))
    actual = list(rr)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert torch.equal(a, e)
