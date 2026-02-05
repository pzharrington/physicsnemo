# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from physicsnemo.core.version_check import check_version_spec

if not check_version_spec("earth2grid", "0.1.0", hard_fail=False):
    pytest.skip(
        "Skipping test because earth2grid is not installed",
        allow_module_level=True,
    )

from physicsnemo.experimental.models.healda import (
    HPXPatchDetokenizer,
    HPXPatchTokenizer,
)
from test import common


def test_hpx_patch_tokenizer_forward(device):
    """Test HPXPatchTokenizer forward pass."""
    torch.manual_seed(0)

    in_channels = 5
    hidden_size = 64
    level_fine = 5
    level_coarse = 3

    model = HPXPatchTokenizer(
        in_channels=in_channels,
        hidden_size=hidden_size,
        level_fine=level_fine,
        level_coarse=level_coarse,
    ).to(device)
    model.eval()

    b, t = 2, 1
    npix = 12 * 4**level_fine
    x = torch.randn(b, in_channels, t, npix, device=device)
    second_of_day = torch.tensor([[43200], [21600]], device=device)
    day_of_year = torch.tensor([[100], [200]], device=device)

    assert common.validate_forward_accuracy(
        model,
        (x, second_of_day, day_of_year),
        file_name="models/healda/data/hpx_tokenizer_output.pth",
        atol=1e-4,
    )


def test_hpx_patch_detokenizer_forward(device):
    """Test HPXPatchDetokenizer forward pass."""
    torch.manual_seed(0)

    hidden_size = 64
    out_channels = 5
    level_coarse = 3
    level_fine = 5
    time_length = 2

    model = HPXPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=time_length,
    ).to(device)
    model.eval()

    b = 2
    L = time_length * 12 * 4**level_coarse
    x = torch.randn(b, L, hidden_size, device=device)
    c = torch.randn(b, hidden_size, device=device)

    assert common.validate_forward_accuracy(
        model,
        (x, c),
        file_name="models/healda/data/hpx_detokenizer_output.pth",
        atol=1e-4,
    )
