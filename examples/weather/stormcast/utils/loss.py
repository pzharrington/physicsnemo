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


"""Loss functions for StormCast training.

Diffusion training uses :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`
with an :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`.
Domain-parallel sigma replication is handled automatically via the
``device_mesh`` parameter of ``MSEDSMLoss`` (which wraps the scheduler with
:class:`~physicsnemo.diffusion.DomainParallelSchedulerWrapper`).
"""

import math
from collections.abc import Sequence

import numpy as np
import torch

from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler


class _EDMNoiseSchedulerLogUniform(EDMNoiseScheduler):
    """EDMNoiseScheduler variant with log-uniform sigma sampling for training.

    Inherits timestep generation, noise addition, and loss weighting from
    ``EDMNoiseScheduler``; only ``sample_time`` differs by drawing sigma
    uniformly in log-space between ``sigma_min`` and ``sigma_max``.
    """

    def sample_time(self, N, *, device=None, dtype=None):
        u = torch.rand(N, device=device, dtype=dtype)
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        return (log_min + u * (log_max - log_min)).exp()


def build_diffusion_loss(
    model: torch.nn.Module,
    loss_cfg,
    device: torch.device,
    logger=None,
    device_mesh=None,
) -> MSEDSMLoss:
    """Create an :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` from Hydra config.

    Parameters
    ----------
    model : torch.nn.Module
        Preconditioned diffusion network.
    loss_cfg : object
        Loss config with ``sigma_distribution``, ``sigma_data``, and
        distribution-specific params (``P_mean``/``P_std`` or
        ``sigma_min``/``sigma_max``).
    device : torch.device
        Device for any tensor sigma_data.
    logger : optional
        Logger for warnings.
    device_mesh : DeviceMesh, optional
        Domain-parallel device mesh.  When provided, sampled sigma values
        are broadcast across the mesh so all spatial shards see the same
        noise level.

    Returns
    -------
    MSEDSMLoss
    """
    sigma_data = loss_cfg.sigma_data
    if isinstance(sigma_data, Sequence):
        sigma_data_tensor = torch.as_tensor(
            list(sigma_data), dtype=torch.float32, device=device
        )[None, :, None, None]
        sigma_data_scalar = float(sigma_data_tensor.mean())
        if logger:
            logger.info(
                f"Per-channel sigma_data detected (mean={sigma_data_scalar:.4f}). "
                "The modern EDMNoiseScheduler uses a scalar sigma_data for loss "
                "weighting. Per-channel support requires a core library extension."
            )
    else:
        sigma_data_scalar = float(sigma_data)

    sigma_dist = loss_cfg.sigma_distribution
    if sigma_dist == "lognormal":
        noise_scheduler = EDMNoiseScheduler(
            sigma_data=sigma_data_scalar,
            P_mean=loss_cfg.P_mean,
            P_std=loss_cfg.P_std,
        )
    elif sigma_dist == "loguniform":
        noise_scheduler = _EDMNoiseSchedulerLogUniform(
            sigma_min=loss_cfg.sigma_min,
            sigma_max=loss_cfg.sigma_max,
            sigma_data=sigma_data_scalar,
        )
    else:
        raise ValueError(
            "training.loss.sigma_distribution must be 'lognormal' or 'loguniform'"
        )

    return MSEDSMLoss(
        model, noise_scheduler, reduction="none", device_mesh=device_mesh
    )


def regression_loss_fn(
    net,
    images: torch.Tensor,
    condition: torch.Tensor,
    class_labels=None,
    lead_time_label: torch.Tensor | None = None,
    augment_pipe=None,
    return_model_outputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MSE loss for the StormCast regression model.

    Shares a compatible call signature with the regression training loop.

    Args:
        net: regression network (e.g. StormCastUNet).
        images: target data, shape ``[B, C, H, W]``.
        condition: model input, shape ``[B, C_cond, H, W]``.
        class_labels: unused (present for call-signature parity).
        lead_time_label: optional lead-time label, shape ``(B,)``.
        augment_pipe: optional data augmentation callable.
        return_model_outputs: if True, return ``(loss, prediction)``.

    Returns:
        Per-pixel squared error ``[B, C, H, W]``, or ``(loss, prediction)``
        when *return_model_outputs* is True.
    """
    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )
    labels = {} if lead_time_label is None else {"lead_time_label": lead_time_label}
    D_yn = net(x=condition, **labels)
    loss = (D_yn - y) ** 2
    if return_model_outputs:
        return loss, D_yn
    return loss


class SigmaBinTracker:
    """Track per-sigma-bin loss and bias for diffusion training diagnostics.

    Accumulates sample-level L2 loss and signed denoising bias into
    equal-probability sigma bins, then logs per-bin means via an
    experiment logger.

    Parameters
    ----------
    loss_cfg : object
        Loss config with attributes: ``track_sigma_bin_loss``, ``sigma_bin_count``,
        ``sigma_bin_edges``, ``sigma_distribution``, ``sigma_min``, ``sigma_max``,
        ``P_mean``, ``P_std``.
    device : torch.device
        Device for accumulator tensors.
    loss_type : str
        ``"regression"`` or ``"edm"``.  Tracking is disabled for regression.
    """

    def __init__(self, loss_cfg, device: torch.device, loss_type: str = "edm"):
        self.enabled = loss_type != "regression" and bool(loss_cfg.track_sigma_bin_loss)
        self.device = device
        self._edges: torch.Tensor | None = None
        self._loss_sum: torch.Tensor | None = None
        self._bias_sum: torch.Tensor | None = None
        self._count: torch.Tensor | None = None
        if not self.enabled:
            return

        if len(loss_cfg.sigma_bin_edges) >= 2:
            edges = np.asarray(loss_cfg.sigma_bin_edges, dtype=np.float64)
        else:
            n_edges = int(loss_cfg.sigma_bin_count) + 1
            if loss_cfg.sigma_distribution == "loguniform":
                q = np.linspace(0.0, 1.0, n_edges, dtype=np.float64)
                log_lo = float(np.log(loss_cfg.sigma_min))
                log_hi = float(np.log(loss_cfg.sigma_max))
                edges = np.exp(log_lo + q * (log_hi - log_lo))
            else:
                q = torch.linspace(0.0, 1.0, n_edges, dtype=torch.float64)
                q = q.clamp(1e-6, 1.0 - 1e-6)
                z = torch.distributions.Normal(0.0, 1.0).icdf(q)
                log_edges = float(loss_cfg.P_mean) + float(loss_cfg.P_std) * z
                edges = torch.exp(log_edges).cpu().numpy()
        self._edges = torch.as_tensor(edges, dtype=torch.float32, device=device)

    @property
    def edges(self) -> list[float] | None:
        """Bin edges as a Python list, or None if disabled."""
        if self._edges is None:
            return None
        return self._edges.detach().cpu().tolist()

    def reset(self) -> None:
        """Zero accumulators at the start of each training step."""
        if not self.enabled:
            return
        n = int(self._edges.numel() - 1)
        self._loss_sum = torch.zeros(n, device=self.device, dtype=torch.float32)
        self._bias_sum = torch.zeros(n, device=self.device, dtype=torch.float32)
        self._count = torch.zeros(n, device=self.device, dtype=torch.float32)

    def update(
        self,
        loss: torch.Tensor,
        sigma: torch.Tensor | None,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one micro-batch of per-sample loss/bias into bins.

        Parameters
        ----------
        loss : torch.Tensor
            Per-pixel loss, shape ``[B, C, H, W]``.
        sigma : torch.Tensor | None
            Sampled sigma values, shape ``[B, 1, 1, 1]`` or ``[B]``.
        bias : torch.Tensor | None
            Per-sample mean signed error ``[B]``.
        """
        if not self.enabled or sigma is None:
            return
        sample_loss = loss.detach().mean(dim=(1, 2, 3))
        sample_sigma = sigma.detach().reshape(-1).to(torch.float32)
        bin_idx = torch.bucketize(sample_sigma, self._edges) - 1
        n_bins = int(self._edges.numel() - 1)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        if not torch.any(valid):
            return
        idx = bin_idx[valid]
        self._loss_sum.index_add_(0, idx, sample_loss[valid])
        self._count.index_add_(
            0, idx, torch.ones_like(sample_loss[valid], dtype=torch.float32)
        )
        if bias is not None:
            self._bias_sum.index_add_(0, idx, bias.detach().to(torch.float32)[valid])

    def log(self, logger, world_size: int = 1) -> None:
        """All-reduce across ranks and log per-bin means.

        Parameters
        ----------
        logger : ExperimentLogger
            Must have a ``log_value(tag, value)`` method.
        world_size : int
            Number of distributed ranks (1 = single GPU).
        """
        if not self.enabled or self._count is None:
            return
        if world_size > 1:
            for t in (self._loss_sum, self._bias_sum, self._count):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        edges = self._edges.detach().cpu().tolist()
        for b in range(int(self._edges.numel() - 1)):
            count = float(self._count[b].item())
            if count <= 0:
                continue
            tag = f"[{edges[b]:.3e},{edges[b + 1]:.3e})"
            logger.log_value(
                f"loss/train_sigma_bin/{tag}",
                float((self._loss_sum[b] / self._count[b]).item()),
            )
            logger.log_value(
                f"bias/train_sigma_bin/{tag}",
                float((self._bias_sum[b] / self._count[b]).item()),
            )
