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

r"""Tests for 2D neighborhood attention on sharded tensors.

This module validates the correctness of :func:`physicsnemo.nn.functional.na2d`
over sharded inputs, covering both forward and backward passes. Sharding is
performed over spatial dimensions (H and/or W), which correspond to
``Shard(1)`` and ``Shard(2)`` in the ``(B, H, W, heads, D)`` layout used by
natten.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from test.conftest import requires_module

from .utils import collective_assert_close, sharded_to_local, validate_shard_tensor_spec


@requires_module("natten")
class TestNA2D:
    """Tests for sharded 2D neighborhood attention."""

    @staticmethod
    def _run_na2d_check(
        distributed_mesh,
        H,
        W,
        num_heads,
        head_dim,
        kernel_size,
        placements,
        backward,
    ):
        from physicsnemo.nn.functional.na2d import na2d

        dm = DistributedManager()

        q = torch.randn(
            1, H, W, num_heads, head_dim, device=dm.device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        sq = scatter_tensor(
            q, 0, distributed_mesh, placements, requires_grad=backward
        )
        sk = scatter_tensor(
            k, 0, distributed_mesh, placements, requires_grad=backward
        )
        sv = scatter_tensor(
            v, 0, distributed_mesh, placements, requires_grad=backward
        )

        # --- Forward: sharded path ---
        d_output = na2d(sq, sk, sv, kernel_size=kernel_size, dilation=1)

        # --- Forward: local reference ---
        if backward:
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)

        ref_output = na2d(q, k, v, kernel_size=kernel_size, dilation=1)

        # Validate spec consistency
        validate_shard_tensor_spec(d_output)

        # Compare forward outputs
        local_output = sharded_to_local(d_output)
        collective_assert_close(
            ref_output,
            local_output,
            atol=1e-4,
            rtol=1e-4,
            msg="na2d forward output mismatch",
        )

        if backward:
            # --- Backward: sharded path ---
            d_output.mean().backward()

            # --- Backward: local reference ---
            ref_output.mean().backward()

            for name, shard_t, ref_t in [
                ("q", sq, q),
                ("k", sk, k),
                ("v", sv, v),
            ]:
                local_grad = sharded_to_local(shard_t.grad)
                collective_assert_close(
                    ref_t.grad,
                    local_grad,
                    atol=1e-3,
                    rtol=1e-3,
                    msg=f"na2d backward {name}.grad mismatch",
                )

    # -- 1D mesh, shard over H (Shard(1)) --

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("H", [16, 32])
    @pytest.mark.parametrize("W", [16])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na2d_1dmesh_shard_h(
        self, distributed_mesh, H, W, num_heads, head_dim, kernel_size, backward
    ):
        self._run_na2d_check(
            distributed_mesh,
            H,
            W,
            num_heads,
            head_dim,
            kernel_size,
            placements=(Shard(1),),
            backward=backward,
        )

    # -- 1D mesh, shard over W (Shard(2)) --

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("H", [16])
    @pytest.mark.parametrize("W", [16, 32])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na2d_1dmesh_shard_w(
        self, distributed_mesh, H, W, num_heads, head_dim, kernel_size, backward
    ):
        self._run_na2d_check(
            distributed_mesh,
            H,
            W,
            num_heads,
            head_dim,
            kernel_size,
            placements=(Shard(2),),
            backward=backward,
        )
