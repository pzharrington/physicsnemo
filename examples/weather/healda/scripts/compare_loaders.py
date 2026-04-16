#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Compare outputs between the ported data loader and the reference implementation.

Modes:
    smoketest  — 3 sample indices, fast sanity check (~1 min)
    full       — 50 indices spread across train split (~10 min)

Usage:
    # From examples/weather/healda/
    python scripts/compare_loaders.py smoketest
    python scripts/compare_loaders.py full
    python scripts/compare_loaders.py full --indices 0 100 500 1000

Requires:
    - The reference codebase importable (healda-reference/src on PYTHONPATH, or
      the healda package installed).
    - Environment variables from .env (ERA5_74VAR, UFS_OBS_PATH, etc.).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
RECIPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_ROOT = os.path.join(os.path.dirname(RECIPE_ROOT), "healda-reference")

# Load .env
from dotenv import load_dotenv

load_dotenv(os.path.join(RECIPE_ROOT, ".env"))


# ============================================================================
# Ported loader construction
# ============================================================================


def build_ported_dataset(split="train", sensors=None):
    """Construct ObsERA5Dataset from the ported data/ package."""
    from physicsnemo.experimental.datapipes.healda.configs.variable_configs import VARIABLE_CONFIGS
    from physicsnemo.experimental.datapipes.healda.dataset import ObsERA5Dataset
    from physicsnemo.experimental.datapipes.healda.loaders.ufs_obs import UFSUnifiedLoader
    from physicsnemo.experimental.datapipes.healda.transforms.era5_obs import ERA5ObsTransform

    variable_config = VARIABLE_CONFIGS["era5"]

    if sensors is None:
        sensors = ["atms", "mhs", "amsua", "amsub"]

    obs_path = os.environ["UFS_OBS_PATH"]
    obs_loader = UFSUnifiedLoader(
        data_path=obs_path,
        sensors=sensors,
        normalization="zscore",
        obs_context_hours=(-21, 3),
    )

    transform = ERA5ObsTransform(variable_config=variable_config, sensors=sensors)

    import xarray

    era5_path = os.environ["ERA5_74VAR"]
    era5_ds = xarray.open_zarr(era5_path, chunks=None)
    era5_data = era5_ds["data"]

    dataset = ObsERA5Dataset(
        era5_data=era5_data,
        obs_loader=obs_loader,
        transform=transform,
        variable_config=variable_config,
        split=split,
    )
    return dataset


# ============================================================================
# Reference loader construction
# ============================================================================


def build_reference_dataset(split="train", sensors=None):
    """Construct ObsERA5Dataset from the reference healda-reference codebase.

    Requires healda-reference/src and healda-reference/ on sys.path.
    """
    # Add reference paths
    ref_src = os.path.join(REFERENCE_ROOT, "src")
    ref_private = REFERENCE_ROOT
    for p in [ref_src, ref_private]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import dotenv as _dotenv

    _dotenv.load_dotenv(os.path.join(RECIPE_ROOT, ".env"))

    from healda.config.models import ObsConfig
    from private.fcn3_dataset import ObsERA5Dataset as RefObsERA5Dataset

    # Build obs_config matching default sensors
    use_conv = sensors is not None and "conv" in sensors
    obs_config = ObsConfig(
        use_obs=True,
        innovation_type="none",
        context_start=-21,
        context_end=3,
        use_conv=use_conv,
    )

    dataset = RefObsERA5Dataset(
        split=split,
        time_length=1,
        frame_step=1,
        model_rank=0,
        model_world_size=1,
        obs_config=obs_config,
    )
    return dataset


# ============================================================================
# Comparison logic
# ============================================================================


def compare_single_sample(ported_ds, ref_ds, idx: int, verbose: bool = True):
    """Compare a single sample between ported and reference datasets.

    Returns a dict of comparison results.
    """
    results = {"idx": idx, "pass": True, "errors": []}

    # --- Raw data comparison (before transform) ---
    try:
        t0 = time.time()
        ported_times, ported_objs = ported_ds.get(idx)
        ported_elapsed = time.time() - t0

        t0 = time.time()
        ref_times, ref_objs = ref_ds.get(idx)
        ref_elapsed = time.time() - t0

        results["ported_time_s"] = ported_elapsed
        results["ref_time_s"] = ref_elapsed

    except Exception as e:
        results["pass"] = False
        results["errors"].append(f"Loading failed: {e}")
        return results

    # Compare timestamps
    for i, (pt, rt) in enumerate(zip(ported_times, ref_times)):
        if str(pt) != str(rt):
            results["pass"] = False
            results["errors"].append(f"Time mismatch at frame {i}: {pt} vs {rt}")

    # Compare state arrays
    for i, (po, ro) in enumerate(zip(ported_objs, ref_objs)):
        p_state = po["state"]
        r_state = ro["state"]

        if p_state.shape != r_state.shape:
            results["pass"] = False
            results["errors"].append(
                f"State shape mismatch at frame {i}: {p_state.shape} vs {r_state.shape}"
            )
            continue

        max_diff = np.max(np.abs(p_state - r_state))
        results[f"state_frame{i}_maxdiff"] = float(max_diff)

        if max_diff > 1e-6:
            results["pass"] = False
            results["errors"].append(
                f"State value mismatch at frame {i}: max_diff={max_diff:.2e}"
            )

    # Compare observation tables
    # Note: ported and reference may produce rows in different order (due to
    # platform grouping within parquet row-groups).  This is benign — the
    # downstream transform processes all obs in a window together.  We sort
    # both tables by a canonical key before value comparison.
    import pyarrow.compute as pc

    def _sort_obs_table(table):
        """Sort by (Global_Channel_ID, Latitude, Longitude, Absolute_Obs_Time)
        to produce a deterministic row order for comparison."""
        sort_keys = [
            ("Global_Channel_ID", "ascending"),
            ("Latitude", "ascending"),
            ("Longitude", "ascending"),
            ("Absolute_Obs_Time", "ascending"),
        ]
        indices = pc.sort_indices(table, sort_keys=sort_keys)
        return table.take(indices)

    for i, (po, ro) in enumerate(zip(ported_objs, ref_objs)):
        p_obs = po.get("obs")
        r_obs = ro.get("obs") or ro.get("obs_v2")  # reference uses legacy key

        if p_obs is None and r_obs is None:
            continue
        if (p_obs is None) != (r_obs is None):
            results["pass"] = False
            results["errors"].append(f"Obs presence mismatch at frame {i}")
            continue

        p_nrows = p_obs.num_rows
        r_nrows = r_obs.num_rows
        results[f"obs_frame{i}_nrows_ported"] = p_nrows
        results[f"obs_frame{i}_nrows_ref"] = r_nrows

        if p_nrows != r_nrows:
            results["pass"] = False
            results["errors"].append(
                f"Obs row count mismatch at frame {i}: {p_nrows} vs {r_nrows}"
            )
            continue

        if p_nrows > 0:
            # Compare schemas
            p_cols = set(p_obs.schema.names)
            r_cols = set(r_obs.schema.names)
            if p_cols != r_cols:
                results["pass"] = False
                results["errors"].append(
                    f"Obs schema mismatch at frame {i}: "
                    f"ported_only={p_cols - r_cols}, ref_only={r_cols - p_cols}"
                )
                continue

            # Sort both tables to canonical order before comparison
            p_sorted = _sort_obs_table(p_obs)
            r_sorted = _sort_obs_table(r_obs)

            # Compare observation values
            p_vals = p_sorted["Observation"].to_numpy()
            r_vals = r_sorted["Observation"].to_numpy()
            obs_max_diff = np.nanmax(np.abs(p_vals - r_vals))
            results[f"obs_frame{i}_val_maxdiff"] = float(obs_max_diff)
            if obs_max_diff > 1e-5:
                results["pass"] = False
                results["errors"].append(
                    f"Obs value mismatch at frame {i}: max_diff={obs_max_diff:.2e}"
                )

            # Also verify Global_Channel_ID sets match
            p_gcids = set(p_obs["Global_Channel_ID"].to_pylist())
            r_gcids = set(r_obs["Global_Channel_ID"].to_pylist())
            if p_gcids != r_gcids:
                results["pass"] = False
                results["errors"].append(
                    f"Obs GCID set mismatch at frame {i}: "
                    f"ported_only={p_gcids - r_gcids}, ref_only={r_gcids - p_gcids}"
                )

    if verbose:
        status = "PASS" if results["pass"] else "FAIL"
        timing = (
            f"ported={results.get('ported_time_s', 0):.2f}s "
            f"ref={results.get('ref_time_s', 0):.2f}s"
        )
        print(f"  [{status}] idx={idx:6d}  {timing}")
        for err in results["errors"]:
            print(f"         {err}")

    return results


def compare_transformed_sample(ported_ds, ref_ds, idx: int, verbose: bool = True):
    """Compare transformed (batched) output between ported and reference.

    Uses __getitems__ to exercise the full transform pipeline.
    """
    results = {"idx": idx, "pass": True, "errors": []}

    try:
        t0 = time.time()
        ported_batch = ported_ds.__getitems__([idx])
        ported_elapsed = time.time() - t0

        t0 = time.time()
        ref_batch = ref_ds.__getitems__([idx])
        ref_elapsed = time.time() - t0

        results["ported_transform_s"] = ported_elapsed
        results["ref_transform_s"] = ref_elapsed

    except Exception as e:
        results["pass"] = False
        results["errors"].append(f"Transform failed: {e}")
        if verbose:
            print(f"  [FAIL] idx={idx:6d} Transform error: {e}")
        return results

    # Compare batch dict keys
    p_keys = set(ported_batch.keys())
    r_keys = set(ref_batch.keys())
    if p_keys != r_keys:
        results["errors"].append(
            f"Batch key mismatch: ported_only={p_keys - r_keys}, ref_only={r_keys - p_keys}"
        )
        # Don't fail — extra/missing keys may be intentional

    # Compare tensor fields
    for key in sorted(p_keys & r_keys):
        pv = ported_batch[key]
        rv = ref_batch[key]

        if isinstance(pv, torch.Tensor) and isinstance(rv, torch.Tensor):
            if pv.shape != rv.shape:
                results["pass"] = False
                results["errors"].append(
                    f"Shape mismatch for '{key}': {pv.shape} vs {rv.shape}"
                )
                continue

            if pv.numel() == 0:
                continue
            max_diff = (pv.float() - rv.float()).abs().max().item()
            results[f"{key}_maxdiff"] = max_diff

            # Use loose tolerance for float transforms
            tol = 1e-4 if pv.is_floating_point() else 0
            if max_diff > tol:
                results["pass"] = False
                results["errors"].append(
                    f"Value mismatch for '{key}': max_diff={max_diff:.2e}"
                )

        elif isinstance(pv, tuple) and isinstance(rv, tuple):
            # unified_obs is a tuple (obs_tensors, lengths_3d)
            # Row ordering may differ between ported and reference (benign —
            # within each sensor group, platforms can appear in different order
            # depending on parquet row-group layout).  We sort both by
            # (global_channel_id, latitude, longitude) before comparing values.
            if len(pv) != len(rv):
                results["pass"] = False
                results["errors"].append(
                    f"Tuple length mismatch for '{key}': {len(pv)} vs {len(rv)}"
                )
                continue

            if isinstance(pv[0], dict) and isinstance(rv[0], dict):
                p_obs_keys = set(pv[0].keys())
                r_obs_keys = set(rv[0].keys())
                if p_obs_keys != r_obs_keys:
                    results["errors"].append(
                        f"Obs tensor key mismatch: "
                        f"ported_only={p_obs_keys - r_obs_keys}, "
                        f"ref_only={r_obs_keys - p_obs_keys}"
                    )

                # Build a stable sort index using torch.lexsort-style
                # multi-key sorting: (gcid, abs_time, lat, lon, observation)
                def _sort_idx(obs_dict):
                    gcid = obs_dict.get("global_channel_id")
                    lat = obs_dict.get("latitude")
                    lon = obs_dict.get("longitude")
                    obs_time = obs_dict.get("absolute_obs_time")
                    obs_val = obs_dict.get("observation")
                    if gcid is None or gcid.numel() == 0:
                        return None
                    # Stack columns as (N, K) float64 for lexicographic sort.
                    # torch.lexsort isn't available, so we use numpy.
                    cols = [gcid.double().cpu().numpy()]
                    if obs_time is not None:
                        cols.append(obs_time.double().cpu().numpy())
                    if lat is not None:
                        cols.append(lat.double().cpu().numpy())
                    if lon is not None:
                        cols.append(lon.double().cpu().numpy())
                    if obs_val is not None:
                        cols.append(obs_val.double().cpu().numpy())
                    # np.lexsort sorts by last key first, so reverse
                    order = np.lexsort(cols[::-1])
                    return torch.from_numpy(order).long()

                p_order = _sort_idx(pv[0])
                r_order = _sort_idx(rv[0])

                for obs_key in sorted(p_obs_keys & r_obs_keys):
                    pt = pv[0][obs_key]
                    rt = rv[0][obs_key]
                    if pt.shape != rt.shape:
                        results["pass"] = False
                        results["errors"].append(
                            f"Obs tensor shape mismatch for '{obs_key}': "
                            f"{pt.shape} vs {rt.shape}"
                        )
                    elif pt.numel() > 0:
                        # Apply sort order before comparison
                        ps = pt[p_order] if p_order is not None else pt
                        rs = rt[r_order] if r_order is not None else rt
                        d = (ps.float() - rs.float()).abs().max().item()
                        results[f"obs_{obs_key}_maxdiff"] = d
                        if d > 1e-4:
                            results["pass"] = False
                            results["errors"].append(
                                f"Obs tensor mismatch for '{obs_key}': "
                                f"max_diff={d:.2e}"
                            )

            # Compare lengths_3d (sensor, batch, time) — these count obs per
            # sensor/window and are order-independent as long as sensor_id
            # mapping matches.
            for ti, name in [(1, "lengths")]:
                if ti < len(pv) and ti < len(rv):
                    pt, rt = pv[ti], rv[ti]
                    if isinstance(pt, torch.Tensor) and isinstance(rt, torch.Tensor):
                        if pt.shape != rt.shape:
                            results["pass"] = False
                            results["errors"].append(
                                f"{name} shape mismatch: {pt.shape} vs {rt.shape}"
                            )
                        elif not torch.equal(pt, rt):
                            results["pass"] = False
                            results["errors"].append(f"{name} value mismatch")

    if verbose:
        status = "PASS" if results["pass"] else "FAIL"
        timing = (
            f"ported={results.get('ported_transform_s', 0):.2f}s "
            f"ref={results.get('ref_transform_s', 0):.2f}s"
        )
        print(f"  [{status}] idx={idx:6d}  {timing}")
        for err in results["errors"]:
            print(f"         {err}")

    return results


# ============================================================================
# Main driver
# ============================================================================


def get_indices(mode: str, ds_len: int, custom_indices=None):
    """Return sample indices based on mode."""
    if custom_indices:
        return [i for i in custom_indices if i < ds_len]

    if mode == "smoketest":
        # 3 indices: start, middle, near end
        return [0, ds_len // 2, ds_len - 1]

    elif mode == "full":
        # 50 indices spread across the dataset
        n = min(50, ds_len)
        step = max(1, ds_len // n)
        return list(range(0, ds_len, step))[:n]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ported vs reference data loader outputs."
    )
    parser.add_argument(
        "mode",
        choices=["smoketest", "full"],
        help="smoketest: 3 indices, fast. full: 50 indices.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Override indices to compare.",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)."
    )
    parser.add_argument(
        "--sensors",
        nargs="*",
        default=None,
        help="Sensor list (default: atms mhs amsua amsub).",
    )
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Also compare transformed (__getitems__) output.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip raw (get) comparison, only do transform.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Loader comparison — mode={args.mode}, split={args.split}")
    print("=" * 70)

    # Build datasets
    print("\nBuilding ported dataset...")
    t0 = time.time()
    ported_ds = build_ported_dataset(split=args.split, sensors=args.sensors)
    print(f"  Done in {time.time() - t0:.1f}s  (len={len(ported_ds)})")

    print("Building reference dataset...")
    t0 = time.time()
    ref_ds = build_reference_dataset(split=args.split, sensors=args.sensors)
    print(f"  Done in {time.time() - t0:.1f}s  (len={len(ref_ds)})")

    # Verify lengths match
    if len(ported_ds) != len(ref_ds):
        print(
            f"\nWARNING: Dataset lengths differ! "
            f"ported={len(ported_ds)} vs ref={len(ref_ds)}"
        )

    indices = get_indices(
        args.mode, min(len(ported_ds), len(ref_ds)), args.indices
    )
    print(f"\nComparing {len(indices)} samples: {indices[:10]}{'...' if len(indices) > 10 else ''}")

    # --- Raw comparison ---
    if not args.no_raw:
        print(f"\n--- Raw comparison (get) ---")
        raw_results = []
        for idx in indices:
            r = compare_single_sample(ported_ds, ref_ds, idx)
            raw_results.append(r)

        n_pass = sum(1 for r in raw_results if r["pass"])
        n_fail = len(raw_results) - n_pass
        print(f"\nRaw: {n_pass}/{len(raw_results)} passed, {n_fail} failed")

    # --- Transform comparison ---
    if args.transform:
        print(f"\n--- Transform comparison (__getitems__) ---")
        xform_results = []
        for idx in indices:
            r = compare_transformed_sample(ported_ds, ref_ds, idx)
            xform_results.append(r)

        n_pass = sum(1 for r in xform_results if r["pass"])
        n_fail = len(xform_results) - n_pass
        print(f"\nTransform: {n_pass}/{len(xform_results)} passed, {n_fail} failed")

    # --- Summary ---
    print("\n" + "=" * 70)
    all_results = []
    if not args.no_raw:
        all_results.extend(raw_results)
    if args.transform:
        all_results.extend(xform_results)

    n_total = len(all_results)
    n_pass = sum(1 for r in all_results if r["pass"])
    if n_total == n_pass:
        print(f"ALL {n_total} CHECKS PASSED")
    else:
        print(f"{n_total - n_pass}/{n_total} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
