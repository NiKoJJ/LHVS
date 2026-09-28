"""
config.py
=========
Central configuration for the Cook/Mertz glacier velocity pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class TemporalMode(Enum):
    MONTHLY   = "monthly"
    SEASONAL  = "seasonal"
    YEARLY    = "yearly"
    ALL       = "all"
    FIXED_6   = "fixed_6"
    FIXED_12  = "fixed_12"
    FIXED_18  = "fixed_18"
    FIXED_30  = "fixed_30"
    FIXED_60  = "fixed_60"


class VxErrorMode(Enum):
    ISOTROPIC    = "isotropic"
    LOCAL_STD    = "local_std"
    PROPORTIONAL = "proportional"


@dataclass
class PipelineConfig:
    # ── Input data directories ────────────────────────────────────────────────
    # Accept one or more directories; all are scanned and merged.
    itslive_dirs: List[str] = field(
        default_factory=lambda: ["/data2/Cook/itslive_pot/v_vx_vy_v_error"]
    )
    gamma_dirs: List[str] = field(
        default_factory=lambda: ["/data2/Cook/gamma_pot/v_vx_vy"]
    )

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir: str = "/data2/Cook/output_velocity"

    # ── Bounding box ──────────────────────────────────────────────────────────
    bbox_latlon: Optional[Tuple[float, float, float, float]] = (
        158.0, -68.5, 163.5, -66.5
    )
    bbox_3031:   Optional[Tuple[float, float, float, float]] = None

    # ── Target grid ───────────────────────────────────────────────────────────
    target_crs:        str   = "EPSG:3031"
    target_resolution: float = 120.0

    # ── Temporal aggregation ──────────────────────────────────────────────────
    temporal_modes: List[TemporalMode] = field(
        default_factory=lambda: [TemporalMode.MONTHLY]
    )

    # ── Data source control ───────────────────────────────────────────────────
    use_itslive:      bool          = True
    use_gamma:        bool          = True
    itslive_max_days: Optional[int] = 33
    gamma_max_days:   Optional[int] = None
    itslive_sensors:  Optional[List[str]] = None

    # ── Error estimation ──────────────────────────────────────────────────────
    vx_error_mode:       VxErrorMode = VxErrorMode.ISOTROPIC
    gamma_error_mode:    VxErrorMode = VxErrorMode.LOCAL_STD
    local_std_window:    int         = 5
    local_std_min_valid: int         = 4

    # ── Coverage filter ───────────────────────────────────────────────────────
    # Fraction of bbox pixels that must be valid for a record to participate.
    # 0.0 = no filter.
    min_coverage_frac: float = 0.0

    # ── Spatial MAD filter (paper Stage 1 — per-record, single-scene) ────────
    # Implements the two-criterion sliding-window filter from:
    #   Remote Sens. 2023, 15, 3079 — Section 2.2 Spatial Filtering.
    # Applied to each scene immediately after read/reproject, before UMT and
    # before the scene enters the temporal stack.
    #
    # Criterion (a): |Vx − median_vx| > MAD_vx  OR  |Vy − median_vy| > MAD_vy
    # Criterion (b): σ_vx > spatial_mad_sigma_thr  OR  σ_vy > spatial_mad_sigma_thr
    #
    # spatial_mad_window   : window side in pixels.
    #                        Default 5 → 2×2 km footprint at 400 m resolution
    #                        (paper: "5×5 moving grid window (2×2 km)").
    # spatial_mad_sigma_thr: threshold for std criterion in m/yr.
    #                        Paper value 150 m/yr (chosen to exceed the expected
    #                        within-2km natural variability of ~100 m/yr).
    # spatial_mad_min_valid: min finite pixels in window to compute statistics;
    #                        sparse-edge pixels are never falsely flagged.
    # spatial_mad_sources  : apply only to the listed data sources.
    use_spatial_mad:         bool  = True
    spatial_mad_window:      int   = 5
    spatial_mad_sigma_thr:   float = 150.0
    spatial_mad_min_valid:   int   = 4
    spatial_mad_tile_rows:   int   = 256
    spatial_mad_sources:     List[str] = field(
        default_factory=lambda: ["itslive", "gamma"]
    )

    # Universal Median Test (single-scene spatial outlier filter)
    # Applied per record before the scene enters the temporal stack.
    use_umt: bool = False
    umt_sources: List[str] = field(
        default_factory=lambda: ["itslive", "gamma"]
    )
    umt_window_size: int = 5
    umt_epsilon: float = 0.1
    umt_threshold: float = 5.0
    umt_min_segment_size: int = 25

    # ── Temporal MAD filter  (per-pixel, within group) ────────────────────────
    # Applied to the (N, H, W) stack BEFORE IQR.
    # For each pixel: flags scenes where |Vx - median_Vx| > k * MAD_Vx
    # (same independently for Vy). Bad pixels are NaN-ed in all bands.
    # Only activated when >= temporal_mad_min_scenes valid obs exist per pixel.
    use_temporal_mad:        bool  = True
    temporal_mad_k:          float = 2.0
    temporal_mad_min_scenes: int   = 3

    # ── IQR outlier removal (Yang et al. 2026) ────────────────────────────────
    use_iqr_filter: bool  = True
    iqr_threshold:  float = 1.5
    iqr_max_cap:    float = 100.0

    # ── Weighted average ──────────────────────────────────────────────────────
    min_valid_obs:   int            = 1
    error_threshold: Optional[float] = None
    no_data_value:   float          = 0.0

    # ── Output / visualisation ────────────────────────────────────────────────
    save_neff:  bool           = True
    save_plots: bool           = True
    plot_dpi:   int            = 150
    plot_vmax:  Optional[float] = None

    # ── Date range filter ─────────────────────────────────────────────────────
    date_start: Optional[str] = None
    date_end:   Optional[str] = None

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_level:  str = "INFO"
    n_workers:  int = 1
    io_threads: int = 4
