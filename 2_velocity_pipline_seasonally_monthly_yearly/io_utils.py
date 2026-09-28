"""
io_utils.py
===========
GeoTIFF output helpers and summary CSV writer.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS

log = logging.getLogger(__name__)

_COMPRESS = "deflate"
_TILE     = {"tiled": True, "blockxsize": 256, "blockysize": 256}


def save_raster(
    filepath:   str | Path,
    data:       np.ndarray,     # (H, W) float32
    transform:  rasterio.Affine,
    crs:        CRS,
    nodata:     float = float("nan"),
    dtype:      str   = "float32",
) -> Path:
    """Write a single-band GeoTIFF."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    arr = data.astype(dtype)
    # Replace NaN with nodata if nodata is finite
    if np.isfinite(nodata):
        arr[~np.isfinite(arr)] = nodata

    profile = dict(
        driver    = "GTiff",
        dtype     = dtype,
        width     = arr.shape[1],
        height    = arr.shape[0],
        count     = 1,
        crs       = crs,
        transform = transform,
        nodata    = nodata,
        compress  = _COMPRESS,
        **_TILE,
    )

    with rasterio.open(str(filepath), "w", **profile) as dst:
        dst.write(arr, 1)

    log.debug("Saved %s", filepath.name)
    return filepath


def save_group_outputs(
    group_name:  str,
    result:      Dict[str, np.ndarray],
    out_subdir:  str | Path,
    transform:   rasterio.Affine,
    crs:         CRS,
    save_neff:   bool = True,
) -> Dict[str, Path]:
    """
    Save all output rasters for one temporal group.

    Files written:
      {group}_vx.tif           – weighted-mean Vx
      {group}_vy.tif           – weighted-mean Vy
      {group}_v.tif            – weighted-mean V (direct)
      {group}_vx_err.tif       – propagated σVx
      {group}_vy_err.tif       – propagated σVy
      {group}_v_err.tif        – propagated σV (direct)
      {group}_v_synth.tif      – synthesised V = √(Vx²+Vy²)
      {group}_v_synth_err.tif  – synthesised σV (error propagation)
      {group}_neff.tif         – effective observation count  [optional]

    Returns dict mapping band key → output path.
    """
    out_dir = Path(out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gn = group_name  # used as filename prefix

    saved: Dict[str, Path] = {}

    bands = {
        "vx":          result.get("vx"),
        "vy":          result.get("vy"),
        "v":           result.get("v"),
        "vx_err":      result.get("vx_err"),
        "vy_err":      result.get("vy_err"),
        "v_err":       result.get("v_err"),
        "v_synth":     result.get("v_synth"),
        "v_synth_err": result.get("v_synth_err"),
    }

    if save_neff:
        bands["neff"] = result.get("neff_v", result.get("neff_vx"))

    for key, arr in bands.items():
        if arr is None:
            continue
        fp = out_dir / f"{gn}_{key}.tif"
        save_raster(fp, arr, transform, crs)
        saved[key] = fp

    log.info("  Saved %d rasters → %s", len(saved), out_dir)
    return saved


def append_summary_row(
    rows:        list,
    group_name:  str,
    result:      Dict[str, np.ndarray],
    n_records:   int,
    mode:        str,
):
    """Append a statistics row (dict) to a running list."""
    v = result.get("v_synth", result.get("v"))
    import numpy as np

    def _stat(arr, pct=None):
        if arr is None or not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.nanpercentile(arr, pct) if pct else np.nanmean(arr))

    cov = (np.sum(np.isfinite(v)) / v.size * 100) if v is not None else float("nan")

    rows.append(dict(
        mode            = mode,
        group           = group_name,
        n_records       = n_records,
        mean_v          = _stat(v),
        mean_vx         = _stat(result.get("vx")),
        mean_vy         = _stat(result.get("vy")),
        mean_v_err      = _stat(result.get("v_synth_err", result.get("v_err"))),
        mean_neff       = _stat(result.get("neff_v")),
        coverage_pct    = round(cov, 2),
    ))
