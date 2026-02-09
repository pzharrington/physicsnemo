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

"""Conditioning embedders for DiT models."""

import math
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn as nn

from physicsnemo.core import Module

from .layers import Linear, PositionalEmbedding


@runtime_checkable
class ConditioningEmbedder(Protocol):
    r"""Protocol for conditioning embedders used in DiT.

    Computes conditioning embedding from timestep and optional additional inputs.
    Implementations must define a forward method and specify their output dimension.

    Forward
    -------
    t : torch.Tensor
        Timestep tensor of shape :math:`(B,)`.
    **kwargs
        Additional conditioning inputs (e.g., condition, class_labels).

    Returns
    -------
    torch.Tensor
        Conditioning embedding of shape :math:`(B, D)` where D is ``output_dim``.
    """

    @property
    def output_dim(self) -> int:
        """Output dimension of conditioning embedding (used for block condition_dim)."""
        ...

    def forward(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute conditioning embedding from timestep and optional inputs."""
        ...


class ZeroConditioningEmbedder(Module):
    r"""Zero conditioning embedder for unconditional/deterministic models (condition_dim=0).

    Returns empty tensors of shape (B, 0) for conditioning, allowing
    AdaLN blocks to operate in bias-only mode (0 x D weight + D bias).

   This is useful when a deterministic model which uses constant timestep/condition values
   is trained using the DiT-style adaptive layer norm mechanism. In this case, the MLP weight matrix
   can be folded into a fixed bias parameter to reduce parameters at inference.
    """

    def __init__(self):
        super().__init__()
        self._output_dim = 0

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.empty(t.shape[0], 0, device=t.device, dtype=t.dtype)


class DiTConditionEmbedder(Module):
    r"""DiT-style conditioning embedder.

    Processes timestep and condition independently, then adds them together at the end.

    Parameters
    ----------
    hidden_size : int
        Output embedding dimension.
    condition_dim : int, optional
        Input condition dimension. If 0, no condition embedding is used.
    max_positions : int, optional
        Maximum positions for positional embedding. Default 10000.
    learnable : bool, optional
        Whether to use learnable MLP after positional embedding. Default True.
    mlp_hidden_dim : int, optional
        Hidden dimension of learnable MLP. Defaults to 2 * hidden_size.
    amp_mode : bool, optional
        Whether mixed-precision (AMP) training is enabled. Default False.

    Forward
    -------
    t : torch.Tensor
        Timestep tensor of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Condition tensor of shape :math:`(B, condition_dim)`.

    Returns
    -------
    torch.Tensor
        Conditioning embedding of shape :math:`(B, hidden_size)`.
    """

    def __init__(
        self,
        hidden_size: int,
        condition_dim: int = 0,
        max_positions: int = 10000,
        learnable: bool = True,
        mlp_hidden_dim: int | None = None,
        amp_mode: bool = False,
    ):
        super().__init__()
        self._output_dim = hidden_size

        self.t_embedder = PositionalEmbedding(
            num_channels=hidden_size,
            max_positions=max_positions,
            learnable=learnable,
            mlp_hidden_dim=mlp_hidden_dim,
            amp_mode=amp_mode,
        )

        self.cond_embedder = (
            Linear(
                in_features=condition_dim,
                out_features=hidden_size,
                bias=False,
                amp_mode=amp_mode,
                init_mode="kaiming_uniform",
                init_weight=0,
                init_bias=0,
            )
            if condition_dim
            else None
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, t: torch.Tensor, condition: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        c = self.t_embedder(t)

        if self.cond_embedder is not None and condition is not None:
            c = c + self.cond_embedder(condition)

        return c


class EDMConditionEmbedder(Module):
    r"""EDM/SongUNet-style conditioning embedder.

    Combines timestep and condition before the MLP.

    Parameters
    ----------
    emb_channels : int
        Output embedding dimension (typically 4 * hidden_size).
    noise_channels : int
        Dimension of positional embedding for the noise/timestep label.
    label_dim : int, optional
        Class label dimension. If 0, no label embedding. Default 0.
    label_dropout : float, optional
        Dropout probability for labels during training. Default 0.0.
    legacy_label_bias : bool, optional
        If ``True`` and ``label_dim`` is 0, add a legacy bias term for backward compatibility.
        Default ``False``.
    max_positions : int, optional
        Maximum positions for positional embedding. Default 10000.

    Forward
    -------
    t : torch.Tensor
        Timestep/noise_labels tensor of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Condition/class labels of shape :math:`(B, label_dim)`.

    Returns
    -------
    torch.Tensor
        Conditioning embedding of shape :math:`(B, emb_channels)`.
    """

    def __init__(
        self,
        emb_channels: int,
        noise_channels: int,
        label_dim: int = 0,
        label_dropout: float = 0.0,
        legacy_label_bias: bool = False,
        max_positions: int = 10000,
        **kwargs,  # Accept and ignore extra kwargs (e.g., amp_mode) for compatibility
    ):
        super().__init__()
        self._output_dim = emb_channels
        self.label_dropout = label_dropout
        self.legacy_label_bias = legacy_label_bias

        self.map_noise = PositionalEmbedding(
            num_channels=noise_channels,
            max_positions=max_positions,
            endpoint=True,
            learnable=False,  # No MLP here - added below
        )

        # Label embedding (added before MLP)
        if label_dim > 0:
            self.map_label = nn.Linear(label_dim, noise_channels)
        elif legacy_label_bias:
            # Preserve legacy bias-only behavior for label_dim=0.
            self.map_label = nn.Linear(0, noise_channels, bias=True)
        else:
            self.map_label = None

        # MLP: Linear → SiLU → Linear (no final SiLU - moved to AdaLN)
        self.map_layer0 = nn.Linear(noise_channels, emb_channels)
        self.map_layer1 = nn.Linear(emb_channels, emb_channels)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, t: torch.Tensor, condition: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        # Positional embedding
        emb = self.map_noise(t)

        # Swap sin/cos order
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)

        # Add label embedding before MLP
        if self.map_label is not None:
            if condition is None and self.legacy_label_bias and self.map_label.in_features == 0:
                emb = emb + self.map_label.bias
            elif condition is not None:
                tmp = condition
                if self.training and self.label_dropout:
                    tmp = tmp * (
                        torch.rand([t.shape[0], 1], device=tmp.device) >= self.label_dropout
                    ).to(tmp.dtype)
                emb = emb + self.map_label(tmp * math.sqrt(self.map_label.in_features))

        # MLP 
        emb = torch.nn.functional.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)

        return emb


def get_conditioning_embedder(
    hidden_size: int,
    conditioning_embedder: Literal["dit", "edm", "zero"] = "dit",
    condition_dim: int = 0,
    **embedder_kwargs: Any,
) -> ConditioningEmbedder:
    r"""Factory function to create conditioning embedders.

    Parameters
    ----------
    hidden_size : int
        The hidden size of the DiT model.
    conditioning_embedder : Literal["dit", "edm", "zero"]
        The type of conditioning embedder to use.
        Options:
            - 'dit': DiT-style, maps timestep and condition independently (late fusion).
            - 'edm': EDM/SongUNet-style, combines timestep and condition before MLP (early fusion).
            - 'zero': Returns empty (B, 0) tensors for bias-only AdaLN (unconditional/ViT-style inference).
    condition_dim : int
        Condition dimension. For 'dit', this is input condition dim.
        For 'edm', this is output emb_channels.
    **embedder_kwargs
        Additional keyword arguments for the embedder.
    """
    if conditioning_embedder == "zero":
        return ZeroConditioningEmbedder()
    if conditioning_embedder == "dit":
        return DiTConditionEmbedder(
            hidden_size=hidden_size,
            condition_dim=condition_dim,
            **embedder_kwargs,
        )
    if conditioning_embedder == "edm":
        return EDMConditionEmbedder(
            emb_channels=condition_dim,
            noise_channels=embedder_kwargs.pop("noise_channels", hidden_size),
            **embedder_kwargs,
        )
    raise ValueError("conditioning_embedder must be 'dit', 'edm', or 'zero'.")
