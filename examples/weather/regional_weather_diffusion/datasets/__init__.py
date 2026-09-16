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

from .data_loader_hrrr_era5 import HrrrEra5Dataset
from .dataset import (
    LoaderSpec,
    StormCastDataset,
    StormCastDataSource,
    default_torch_loader,
    flatten_group_collate,
)
from .mock import MockDataset

# StormCastDataSource implementations keyed by "<module>.<ClassName>", matching
# the `dataset.name` config field.
dataset_classes: dict[str, type[StormCastDataSource]] = {
    "data_loader_hrrr_era5.HrrrEra5Dataset": HrrrEra5Dataset,
    "mock.MockDataset": MockDataset,
}

__all__ = [
    "HrrrEra5Dataset",
    "LoaderSpec",
    "MockDataset",
    "StormCastDataSource",
    "StormCastDataset",
    "dataset_classes",
    "default_torch_loader",
    "flatten_group_collate",
]
