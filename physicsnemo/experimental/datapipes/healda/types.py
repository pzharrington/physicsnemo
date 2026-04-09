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
"""Core data types for HealDA data loading."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from enum import Enum
from typing import Any, Optional, TypedDict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Time unit enum
# ---------------------------------------------------------------------------


class TimeUnit(Enum):
    """Time units supported by the dataset.

    Values are the pandas frequency strings (offset aliases).
    """

    HOUR = "h"
    DAY = "D"
    MINUTE = "min"
    SECOND = "s"

    def to_timedelta(self, steps: float) -> timedelta:
        return {
            TimeUnit.HOUR: timedelta(hours=steps),
            TimeUnit.DAY: timedelta(days=steps),
            TimeUnit.MINUTE: timedelta(minutes=steps),
            TimeUnit.SECOND: timedelta(seconds=steps),
        }[self]


# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VariableConfig:
    """Describes the variables and pressure levels for a dataset."""

    name: str
    variables_2d: list[str]
    variables_3d: list[str]
    levels: list[int]
    variables_static: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Batch info (normalization metadata)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BatchInfo:
    """Metadata about batch channels and normalization constants."""

    channels: list[str]
    time_step: int = 1  # Time (in units ``time_unit``) between consecutive frames
    time_unit: TimeUnit = TimeUnit.HOUR
    scales: Any | None = None
    center: Any | None = None

    def __post_init__(self):
        if isinstance(self.time_unit, str):
            raise ValueError("Time unit is a str. Should be a TimeUnit.")

    @staticmethod
    def loads(s):
        kw = json.loads(s)
        if "time_unit" in kw:
            kw["time_unit"] = TimeUnit(kw["time_unit"])
        kw.pop("residual_normalization", None)
        return BatchInfo(**kw)

    def asdict(self):
        out = {}
        out["channels"] = self.channels
        out["time_step"] = self.time_step
        out["time_unit"] = self.time_unit.value
        if self.scales is not None:
            out["scales"] = np.asarray(self.scales).tolist()
        else:
            out["scales"] = None
        if self.center is not None:
            out["center"] = np.asarray(self.center).tolist()
        else:
            out["center"] = None
        return out

    def sel_channels(self, channels: list[str]):
        channels = list(channels)
        index = np.array([self.channels.index(ch) for ch in channels])
        scales = None
        if self.scales is not None:
            scales = np.asarray(self.scales)[index]
        center = None
        if self.center is not None:
            center = np.asarray(self.center)[index]
        return BatchInfo(
            time_step=self.time_step,
            time_unit=self.time_unit,
            channels=channels,
            scales=scales,
            center=center,
        )

    def denormalize(self, x):
        scales = torch.as_tensor(self.scales).to(x)
        scales = scales.view(-1, 1, 1)
        center = torch.as_tensor(self.center).to(x)
        center = center.view(-1, 1, 1)
        return x * scales + center

    def get_time_delta(self, t: int) -> timedelta:
        """Get time offset of the *t*-th frame in a frame sequence."""
        total_steps = t * self.time_step
        return self.time_unit.to_timedelta(total_steps)


# ---------------------------------------------------------------------------
# Unified observation structure
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UnifiedObservation:
    """Unified observation structure for both satellite and conventional observations."""

    obs: torch.Tensor  # (n_obs,) observation values
    time: torch.Tensor  # (n_obs,) timestamps in ns since epoch
    float_metadata: torch.Tensor  # (n_obs, n_features)

    # Integer metadata (each shape (n_obs,))
    pix: torch.Tensor  # HEALPix pixel index (NEST)
    local_channel: torch.Tensor
    platform: torch.Tensor
    obs_type: torch.Tensor
    global_channel: torch.Tensor

    hpx_level: int  # HEALPix level that ``pix`` is defined at

    lengths: torch.Tensor | None = (
        None  # 3D: (n_active_sensors, batch, time) per-window obs counts
    )
    sensor_id_to_local: torch.Tensor | None = (
        None  # (max_sensor_id+1,) map: sensor_id -> local_idx (-1 if inactive)
    )

    @classmethod
    def empty(
        cls,
        device: str = "cpu",
        hpx_level: int = 8,
        batch_dims: tuple[int, int] = (1, 1),
    ) -> UnifiedObservation:
        B, T = batch_dims
        return cls(
            obs=torch.empty(0, device=device),
            time=torch.empty(0, dtype=torch.long, device=device),
            float_metadata=torch.empty((0, 28), device=device),
            pix=torch.empty(0, dtype=torch.long, device=device),
            local_channel=torch.empty(0, dtype=torch.long, device=device),
            platform=torch.empty(0, dtype=torch.long, device=device),
            obs_type=torch.empty(0, dtype=torch.long, device=device),
            global_channel=torch.empty(0, dtype=torch.long, device=device),
            lengths=torch.zeros(1, B, T, dtype=torch.long, device=device),
            sensor_id_to_local=torch.zeros(1, dtype=torch.long, device=device),
            hpx_level=hpx_level,
        )

    @property
    def batch_dims(self):
        """Return ``(batch, time)`` shape from 3D offsets ``(S, B, T)``."""
        if self.lengths is not None:
            return self.lengths.shape[-2:]
        else:
            return ()

    def __repr__(self):
        nobs = self.obs.shape[0]
        return f"UnifiedObservation({nobs=}, batch_dims={self.batch_dims})"

    def to(self, device=None, dtype=None, non_blocking=True):
        """Move all tensors to *device* and/or convert *dtype*."""

        def _move(x):
            if x is None:
                return None
            return x.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return UnifiedObservation(
            obs=_move(self.obs),
            time=_move(self.time),
            float_metadata=_move(self.float_metadata),
            pix=_move(self.pix),
            local_channel=_move(self.local_channel),
            platform=_move(self.platform),
            obs_type=_move(self.obs_type),
            global_channel=_move(self.global_channel),
            hpx_level=self.hpx_level,
            lengths=_move(self.lengths),
            sensor_id_to_local=_move(self.sensor_id_to_local),
        )


# ---------------------------------------------------------------------------
# Batch TypedDict
# ---------------------------------------------------------------------------


class Batch(TypedDict):
    """A batch of model inputs produced by the data pipeline."""

    target: torch.Tensor  # (b, c, t, x)
    condition: torch.Tensor  # (b, c_cond, t, x)
    second_of_day: torch.Tensor  # (b, t)
    day_of_year: torch.Tensor  # (b, t)
    labels: torch.Tensor  # (b, num_classes)
    timestamp: torch.Tensor  # (b, t)
    unified_obs: Optional[UnifiedObservation]


def empty_batch(
    *,
    batch_gpu: int,
    out_channels: int,
    condition_channels: int,
    time_length: int,
    x_size: int,
    device: torch.device | str,
) -> Batch:
    """Create an empty batch with the given dimensions."""
    if x_size <= 0:
        raise ValueError(f"x_size must be positive, got {x_size}")

    return {
        "target": torch.empty(
            [batch_gpu, out_channels, time_length, x_size], device=device
        ),
        "condition": torch.empty(
            [batch_gpu, condition_channels, time_length, x_size], device=device
        ),
        "second_of_day": torch.empty([batch_gpu, time_length], device=device),
        "day_of_year": torch.empty([batch_gpu, time_length], device=device),
        "labels": torch.empty([batch_gpu, 0], device=device),
        "timestamp": torch.empty(
            [batch_gpu, time_length], dtype=torch.long, device=device
        ),
        "unified_obs": UnifiedObservation.empty(
            device=device, batch_dims=(batch_gpu, time_length)
        ),
    }


# ---------------------------------------------------------------------------
# Sensor-level splitting
# ---------------------------------------------------------------------------


@torch.compiler.disable
def split_by_sensor(
    obs: UnifiedObservation, target_sensor_ids: list[int]
) -> dict[int, UnifiedObservation]:
    """Slice a ``UnifiedObservation`` into per-sensor sub-objects.

    Uses precomputed ``lengths`` and ``sensor_id_to_local`` for efficient
    splitting without per-element sensor ID checks.
    """
    if obs.lengths is None or obs.sensor_id_to_local is None:
        raise ValueError("lengths is required for split_by_sensor")

    lengths = obs.lengths  # [S, B, T]
    sensor_id_to_local = obs.sensor_id_to_local

    device = obs.obs.device
    B, T = obs.batch_dims

    sizes = lengths.sum(dim=(1, 2)).tolist()
    obs_fields = [
        obs.obs,
        obs.time,
        obs.float_metadata,
        obs.pix,
        obs.local_channel,
        obs.platform,
        obs.obs_type,
        obs.global_channel,
    ]
    splits = [torch.split(f, sizes) for f in obs_fields]

    out = {}
    for sensor_id in target_sensor_ids:
        if sensor_id < len(sensor_id_to_local):
            s_local = int(sensor_id_to_local[sensor_id].item())
        else:
            s_local = -1

        single_sensor_map = torch.full(
            (sensor_id + 1,), -1, dtype=torch.int32, device=device
        )
        single_sensor_map[sensor_id] = 0

        if s_local < 0:
            sensor_lengths = torch.zeros((1, B, T), dtype=lengths.dtype, device=device)
            out[sensor_id] = UnifiedObservation(
                obs=obs.obs[:0],
                time=obs.time[:0],
                float_metadata=obs.float_metadata[:0],
                pix=obs.pix[:0],
                local_channel=obs.local_channel[:0],
                platform=obs.platform[:0],
                obs_type=obs.obs_type[:0],
                global_channel=obs.global_channel[:0],
                hpx_level=obs.hpx_level,
                lengths=sensor_lengths,
                sensor_id_to_local=single_sensor_map,
            )
        else:
            out[sensor_id] = UnifiedObservation(
                obs=splits[0][s_local],
                time=splits[1][s_local],
                float_metadata=splits[2][s_local],
                pix=splits[3][s_local],
                local_channel=splits[4][s_local],
                platform=splits[5][s_local],
                obs_type=splits[6][s_local],
                global_channel=splits[7][s_local],
                hpx_level=obs.hpx_level,
                lengths=lengths[s_local : s_local + 1],
                sensor_id_to_local=single_sensor_map,
            )

    return out
