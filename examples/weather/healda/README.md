# HealDA — AI-based Data Assimilation on the HEALPix Grid

> **This recipe is under active construction.**
> Structure and functionality are subject to changes.

HealDA is a stateless assimilation model that produces a single
global weather analysis from conventional and satellite
observations. It operates on a HEALPix level-6 padded XY grid
and outputs ERA5-compatible atmospheric variables.

This example provides a recipe to train HealDA, with support
for extension to custom data.

## Setup

Start by installing PhysicsNeMo (if not already installed) with
the `healda` optional dependency group, along with the packages
in `requirements.txt`. Then, copy this folder
(`examples/weather/healda`) to a system with a GPU available.
Also, prepare a dataset that can serve training data according
to the protocols outlined in the
[Generalized Data Loading](#generalized-data-loading) section
below.

## Generalized Data Loading

The `physicsnemo.experimental.datapipes.healda` package provides
a composable data loading pipeline with clear extension points.
The architecture separates components into loaders, transforms,
datasets, and sampling infrastructure.

### Architecture

```text
ObsERA5Dataset(era5_data, obs_loader, transform)
  |  Temporal windowing via FrameIndexGenerator
  |  __getitems__ -> get() per index -> transform.transform()
  v
ChunkedDistributedSampler (contiguous chunks for cache locality)
  |
DataLoader (1 worker each, pin_memory, persistent_workers)
  |
RoundRobinLoader (interleaves per-worker DataLoaders)
  |
prefetch_map(loader, transform.device_transform)
  |
Training loop (GPU-ready batch)
```

### Key Protocols

Custom data sources and transforms plug in via these protocols
(see `physicsnemo.experimental.datapipes.healda.protocols`):

**`ObsLoader`** — the observation loading interface:

```python
class MyObsLoader:
    async def sel_time(self, times):
        """Return {"obs": [pa.Table, ...]}"""
        ...
```

**`Transform`** / **`DeviceTransform`** — two-stage batch
processing:

```python
class MyTransform:
    def transform(self, times, frames):
        """CPU-side: normalize, encode obs, time features."""
        ...

    def device_transform(self, batch, device):
        """GPU-side: move to device, compute obs features."""
        ...
```

### Provided Implementations

| Component | Module | Description |
|---|---|---|
| `ObsERA5Dataset` | `dataset` | ERA5 state + observations |
| `UFSUnifiedLoader` | `loaders.ufs_obs` | Parquet obs loader |
| `ERA5Loader` | `loaders.era5` | Async ERA5 zarr loader |
| `ERA5ObsTransform` | `transforms.era5_obs` | Two-stage transform |
| `ChunkedDistributedSampler` | `samplers` | Distributed sampler |
| `RoundRobinLoader` | `samplers` | Multi-loader interleave |
| `prefetch_map` | `prefetch` | CUDA stream prefetching |

All modules above are under
`physicsnemo.experimental.datapipes.healda`.

### Writing a Custom Observation Loader

Implement `async def sel_time(times)` returning a dict with
observation data per timestamp:

```python
class GOESRadianceLoader:
    def __init__(self, data_path, channels):
        self.data_path = data_path
        self.channels = channels

    async def sel_time(self, times):
        tables = []
        for t in times:
            table = self._load_goes_radiances(t)
            tables.append(table)
        return {"obs": tables}
```

Then pass it to the dataset:

```python
from physicsnemo.experimental.datapipes.healda import (
    ObsERA5Dataset,
)
from physicsnemo.experimental.datapipes.healda.transforms.era5_obs import (
    ERA5ObsTransform,
)
from physicsnemo.experimental.datapipes.healda.configs.variable_configs import (
    VARIABLE_CONFIGS,
)

dataset = ObsERA5Dataset(
    era5_data=era5_xr["data"],
    obs_loader=GOESRadianceLoader(...),
    transform=ERA5ObsTransform(...),
    variable_config=VARIABLE_CONFIGS["era5"],
)
```
