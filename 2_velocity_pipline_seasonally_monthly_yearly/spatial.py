"""
spatial.py  —  clip-first, sentinel-safe reprojection

Root-cause of tile-boundary artifacts (fixed here):
  1. Using NaN as src_nodata in bilinear reproject is CONSERVATIVE:
     any of the 4 kernel neighbours being NaN → output NaN.
     This erodes the valid-data extent inward ~1 source pixel, making
     individual satellite footprints visible in the composite.
  FIX: use a numeric sentinel (-9999) as nodata during reproject;
       only convert to NaN afterwards.

  2. rio_mask gives a cleaner polygon boundary than a simple window read:
     pixels outside the bbox geometry are set to a single nodata value,
     not left as irregular ITS_LIVE fill patterns.
  FIX: use rio_mask (same as old glacier_velocity_pipeline.py).
"""
from __future__ import annotations
import logging
from typing import Tuple, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import box, mapping

import pyproj
from shapely.ops import transform as shp_transform

log = logging.getLogger(__name__)

# Numeric sentinel used during bilinear reprojection.
# Must NOT appear as a real velocity value; -9999 is safe for m/yr.
_SENTINEL = np.float32(-9999.0)


# ── bbox helpers ─────────────────────────────────────────────────────────────

def resolve_bbox_3031(cfg) -> Tuple[float, float, float, float]:
    if cfg.bbox_3031 is not None:
        return tuple(cfg.bbox_3031)
    if cfg.bbox_latlon is not None:
        lon_min, lat_min, lon_max, lat_max = cfg.bbox_latlon
        x_min, y_min, x_max, y_max = transform_bounds(
            CRS.from_epsg(4326), CRS.from_string(cfg.target_crs),
            lon_min, lat_min, lon_max, lat_max, densify_pts=21)
        log.info("BBox (lat/lon): lon=[%.3f,%.3f] lat=[%.3f,%.3f]  ->  "
                 "EPSG:3031 x=[%.0f,%.0f] y=[%.0f,%.0f]",
                 lon_min, lon_max, lat_min, lat_max,
                 x_min, x_max, y_min, y_max)
        return x_min, y_min, x_max, y_max
    raise ValueError("Neither bbox_3031 nor bbox_latlon set in config.")


def build_target_grid(bbox_3031, resolution, crs_str="EPSG:3031"):
    x_min, y_min, x_max, y_max = bbox_3031
    crs    = CRS.from_string(crs_str)
    width  = max(1, int(round((x_max - x_min) / resolution)))
    height = max(1, int(round((y_max - y_min) / resolution)))
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    log.info("Target grid: %d x %d px  res=%.0f m  "
             "bbox=[%.0f,%.0f,%.0f,%.0f]  CRS=%s",
             width, height, resolution, x_min, y_min, x_max, y_max, crs_str)
    return transform, width, height, crs


# ── geometry reprojection helper ─────────────────────────────────────────────

def _reproject_geom(geom, src_crs_str, dst_crs_str):
    tr = pyproj.Transformer.from_crs(src_crs_str, dst_crs_str, always_xy=True)
    return shp_transform(tr.transform, geom)


# ── core reader ───────────────────────────────────────────────────────────────

def read_and_reproject(
    filepath,
    target_transform,
    target_width:  int,
    target_height: int,
    target_crs,
    no_data_value: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Clip-first, sentinel-safe read + reproject.

    Mirrors old glacier_velocity_pipeline.py:
      1. rio_mask(crop=True) clips to bbox polygon in source CRS.
      2. All invalid pixels replaced with numeric _SENTINEL (not NaN).
      3. Reproject small clip with src_nodata=_SENTINEL.
         -> no extra edge erosion from NaN conservatism.
      4. Convert _SENTINEL -> NaN after reproject.
    """
    try:
        with rasterio.open(str(filepath)) as src:
            src_crs = src.crs or target_crs

            # Build bbox polygon in source CRS
            dst_bounds = array_bounds(target_height, target_width, target_transform)
            bbox_geom  = box(*dst_bounds)   # in target_crs
            if src_crs.to_epsg() != target_crs.to_epsg():
                clip_geom = _reproject_geom(
                    bbox_geom, target_crs.to_string(), src_crs.to_string())
            else:
                clip_geom = bbox_geom

            nd = src.nodata if src.nodata is not None else no_data_value
            try:
                clipped, clip_tf = rio_mask(
                    src, [mapping(clip_geom)],
                    crop=True, filled=True, nodata=nd)
            except Exception:
                return None

            arr = clipped[0].astype(np.float32)

            # Replace all invalid pixels with numeric sentinel
            if src.nodata is not None:
                arr[arr == float(src.nodata)] = _SENTINEL
            arr[arr == float(no_data_value)] = _SENTINEL
            arr[~np.isfinite(arr)]           = _SENTINEL

            if np.all(arr == _SENTINEL):
                return None

        # Reproject small clipped array to fixed target grid.
        # src_nodata = _SENTINEL (numeric) prevents NaN-bleed at tile edges.
        dst = np.full((target_height, target_width), _SENTINEL, dtype=np.float32)
        reproject(
            source=arr,
            destination=dst,
            src_transform=clip_tf,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=float(_SENTINEL),
            dst_nodata=float(_SENTINEL),
        )

        dst[dst == _SENTINEL]  = np.nan
        dst[~np.isfinite(dst)] = np.nan

        if np.all(np.isnan(dst)):
            return None
        return dst

    except Exception as exc:
        log.error("Failed to read %s : %s", filepath, exc)
        return None


def read_triplet(record, target_transform, target_width, target_height,
                 target_crs, no_data_value=0.0) -> Optional[dict]:
    kw = dict(target_transform=target_transform,
              target_width=target_width, target_height=target_height,
              target_crs=target_crs, no_data_value=no_data_value)

    vx = read_and_reproject(record.vx_path, **kw)
    if vx is None:
        return None

    vy      = read_and_reproject(record.vy_path, **kw)
    v       = read_and_reproject(record.v_path,  **kw)
    v_error = None
    if record.v_error_path:
        v_error = read_and_reproject(record.v_error_path, **kw)
        if v_error is not None:
            v_error[v_error <= 0] = np.nan

    return {"vx": vx, "vy": vy, "v": v, "v_error": v_error}
