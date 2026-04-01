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

"""Domain-parallel utilities for diffusion training and sampling."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager


class DomainParallelSchedulerWrapper:
    r"""Wrap a noise scheduler for domain-parallel diffusion training and sampling.

    In domain-parallel diffusion the spatial domain is split across multiple
    ranks.  This wrapper ensures that tensors produced by the scheduler are
    distributed correctly across the domain mesh:

    * :meth:`sample_time` — broadcasts sampled times so every shard sees the
      same noise level per batch element (training).
    * :meth:`timesteps` — returns a *replicated* tensor on the domain mesh so
      that solver arithmetic with sharded latents is type-compatible
      (sampling).
    * :meth:`init_latents` — returns a *sharded* tensor on the domain mesh,
      split along the chosen spatial dimension (sampling).

    All other scheduler methods (``add_noise``, ``loss_weight``,
    ``get_denoiser``, etc.) are delegated unchanged to the wrapped scheduler.

    Parameters
    ----------
    scheduler : NoiseScheduler
        The inner noise scheduler to wrap.  Any object that implements the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol.
    device_mesh : DeviceMesh
        The device mesh defining the domain-parallel group.
    shard_dim : int
        The tensor dimension along which :meth:`init_latents` shards the
        initial latent state.  For example, for ``(B, C, H, W)`` data sharded
        along the height axis, use ``shard_dim=2``.
    """

    def __init__(
        self,
        scheduler: object,
        device_mesh: DeviceMesh,
        shard_dim: int,
    ) -> None:
        self._inner = scheduler
        self._mesh = device_mesh
        self._shard_dim = shard_dim
        dm = DistributedManager()
        self._group = dm.get_mesh_group(device_mesh)
        self._source_rank = dist.get_global_rank(self._group, 0)

    @property
    def inner_scheduler(self) -> object:
        """The wrapped noise scheduler."""
        return self._inner

    @property
    def device_mesh(self) -> DeviceMesh:
        """The device mesh used for broadcasting."""
        return self._mesh

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""Sample diffusion times and broadcast across the domain-parallel group.

        Rank 0 of the mesh group draws ``N`` random times from the inner
        scheduler; all other ranks receive the same values via broadcast.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`, identical on all
            ranks within the domain-parallel group.
        """
        t = self._inner.sample_time(N, device=device, dtype=dtype)
        return self._broadcast_time(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N_plus_1"]:
        r"""Generate time-steps replicated across the domain-parallel group.

        The inner scheduler produces a plain 1-D tensor of time-steps.  This
        method wraps it as a *replicated* :class:`ShardTensor` on the domain
        mesh so that solver arithmetic with sharded latents is type-compatible.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Replicated time-steps tensor of shape :math:`(N + 1,)`.
        """
        t = self._inner.timesteps(num_steps, device=device, dtype=dtype)
        return self._scatter(t, placements=(Replicate(),))

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""Initialize latent state sharded across the domain-parallel group.

        Rank 0 generates the full initial noise via the inner scheduler's
        :meth:`init_latents`, then scatters it across the domain mesh with
        ``Shard(shard_dim)`` placement.

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g. ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sharded initial noisy latent of shape :math:`(B, *spatial\_shape)`.
        """
        # Unwrap tN to a plain tensor if it is a DTensor/ShardTensor
        # (e.g. from timesteps() which returns Replicate placement),
        # because the inner scheduler operates on plain tensors.
        if hasattr(tN, "to_local"):
            tN = tN.to_local()
        xN = self._inner.init_latents(spatial_shape, tN, device=device, dtype=dtype)
        return self._scatter(xN, placements=(Shard(self._shard_dim),))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _broadcast_time(self, t: torch.Tensor) -> torch.Tensor:
        """Broadcast *t* from rank 0 of the domain group to all other ranks."""
        dist.broadcast(t, src=self._source_rank, group=self._group)
        return t

    def _scatter(
        self,
        tensor: torch.Tensor,
        placements: tuple,
    ) -> torch.Tensor:
        """Scatter *tensor* from rank 0 across the domain mesh."""
        from physicsnemo.domain_parallel.shard_tensor import scatter_tensor

        return scatter_tensor(
            tensor,
            self._source_rank,
            self._mesh,
            placements=placements,
            global_shape=tensor.shape,
            dtype=tensor.dtype,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)
