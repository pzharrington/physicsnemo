# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
This file is meant to provide a compatibility layer for physicsnemo v1

You can do
```
>>> import physicsnemo.compat as physicsnemo
>>> # All previous paths should work.

```
"""

import importlib
import sys
import warnings

COMPAT_MAP = {
    "physicsnemo.utils.filesystem": "physicsnemo.core.filesystem",
    "physicsnemo.utils.version_check": "physicsnemo.core.version_check",
    "physicsnemo.models.meta": "physicsnemo.core.meta",
    "physicsnemo.models.module": "physicsnemo.core.module",
    "physicsnemo.utils.neighbors": "physicsnemo.nn.neighbors",
    "physicsnemo.utils.sdf": "physicsnemo.nn.sdf",
    "physicsnemo.models.layers": "physicsnemo.nn",
    # "physicsnemo.models.layers.activations": "physicsnemo.nn.activations",
    # "physicsnemo.models.layers.attention_layers": "physicsnemo.nn.layers.attention_layers",
    # "physicsnemo.models.layers.ball_query": "physicsnemo.nn.layers.ball_query",
    # "physicsnemo.models.layers.conv_layers": "physicsnemo.nn.layers.conv_layers",
    # "physicsnemo.models.layers.dgm_layers": "physicsnemo.nn.layers.dgm_layers",
    # "physicsnemo.models.layers.drop": "physicsnemo.nn.layers.drop",
    # "physicsnemo.models.layers.fft": "physicsnemo.nn.layers.fft",
    # "physicsnemo.models.layers.fourier_layers": "physicsnemo.nn.layers.fourier_layers",
    # "physicsnemo.models.layers.fully_connected_layers": "physicsnemo.nn.layers.fully_connected_layers",
    # "physicsnemo.models.layers.fused_silu": "physicsnemo.nn.layers.fused_silu",
    # "physicsnemo.models.layers.interpolation": "physicsnemo.nn.layers.interpolation",
    # "physicsnemo.models.layers.kan_layers": "physicsnemo.nn.layers.kan_layers",
    # "physicsnemo.models.layers.mlp_layers": "physicsnemo.nn.layers.mlp_layers",
    # "physicsnemo.models.layers.resample_layers": "physicsnemo.nn.layers.resample_layers",
    # "physicsnemo.models.layers.siren_layers": "physicsnemo.nn.layers.siren_layers",
    # "physicsnemo.models.layers.spectral_layers": "physicsnemo.nn.layers.spectral_layers",
    # "physicsnemo.models.layers.transfomer_decoder": "physicsnemo.nn.layers.transfomer_decoder",
    # "physicsnemo.models.layers.transformer_layers": "physicsnemo.nn.layers.transformer_layers",
    # "physicsnemo.models.layers.weight_fact": "physicsnemo.nn.layers.weight_fact",
    # "physicsnemo.models.layers.weight_norm": "physicsnemo.nn.layers.weight_norm",
    # "physicsnemo.utils.graphcast": "physicsnemo.models.graphcast.utils",
    "physicsnemo.utils.graphcast": "physicsnemo.models.graphcast.utils",
    "physicsnemo.utils.diffusion": "physicsnemo.models.diffusion.utils",
    "physicsnemo.utils.patching": "physicsnemo.nn.patching",
    "physicsnemo.utils.domino": "physicsnemo.models.domino.utils",
    "physicsnemo.launch.utils.checkpoint": "physicsnemo.utils.checkpoint",
    "physicsnemo.launch.logging": "physicsnemo.utils.logging",
}


def install():
    """Install backward-compatibility shims."""
    for old_name, new_name in COMPAT_MAP.items():
        try:
            new_mod = importlib.import_module(new_name)
        except ImportError:
            warnings.warn(
                f"Failed to import new module '{new_name}' for compat alias '{old_name}'"
            )
            continue

        # Register module alias
        sys.modules[old_name] = new_mod

        # Attach the alias on the parent package so "from pkg.subpkg import name" works
        try:
            parent_name, child = old_name.rsplit(".", 1)
            parent_mod = sys.modules.get(parent_name) or importlib.import_module(
                parent_name
            )
            setattr(parent_mod, child, new_mod)
        except Exception:
            warnings.warn(
                f"Failed to attach '{old_name}' onto its parent for compat alias; using sys.modules only"
            )

        warnings.warn(
            f"[compat] {old_name} is moved; use {new_name} instead",
            DeprecationWarning,
        )
