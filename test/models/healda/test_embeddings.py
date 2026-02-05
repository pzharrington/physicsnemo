# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.experimental.models.healda import CalendarEmbedding, FrequencyEmbedding
from test import common


def test_frequency_embedding_forward(device):
    """Test FrequencyEmbedding forward pass."""
    torch.manual_seed(0)

    num_channels = 8
    model = FrequencyEmbedding(num_channels=num_channels).to(device)
    model.eval()

    inp = torch.randn(2, 3, 50, device=device)

    assert common.validate_forward_accuracy(
        model,
        (inp,),
        file_name="models/healda/data/frequency_embedding_output.pth",
        atol=1e-5,
    )


def test_calendar_embedding_forward(device):
    """Test CalendarEmbedding forward pass."""
    torch.manual_seed(0)

    npix = 50
    embed_channels = 4
    lon = torch.linspace(-180, 180, npix, device=device)
    model = CalendarEmbedding(lon=lon, embed_channels=embed_channels).to(device)
    model.eval()

    day_of_year = torch.tensor([[100, 150, 200], [50, 100, 150]], device=device)
    second_of_day = torch.tensor(
        [[43200, 21600, 64800], [0, 43200, 86399]], device=device
    )

    assert common.validate_forward_accuracy(
        model,
        (day_of_year, second_of_day),
        file_name="models/healda/data/calendar_embedding_output.pth",
        atol=1e-5,
    )


def test_calendar_embedding_shape_mismatch():
    """Test CalendarEmbedding raises on shape mismatch."""
    lon = torch.linspace(-180, 180, 10)
    model = CalendarEmbedding(lon=lon, embed_channels=4)

    day_of_year = torch.tensor([[100, 101]])
    second_of_day = torch.tensor([[43200]])

    with pytest.raises(ValueError):
        model(day_of_year=day_of_year, second_of_day=second_of_day)
