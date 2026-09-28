#!/usr/bin/env python3
"""
run_pipeline.py
===============
Command-line entry point for the Cook/Mertz glacier velocity pipeline.

Examples
--------
# Monthly average, Cook Ice Shelf region (WGS-84 bbox), both data sources
python run_pipeline.py \
    --itslive-dir  /data2/Cook/itslive_pot/v_vx_vy_v_error \
    --gamma-dir    /data2/Cook/gamma_pot/v_vx_vy \
    --output-dir   /data2/Cook/output_velocity \
    --modes        monthly \
    --bbox-latlon  158.0 -68.5 163.5 -66.5 \
    --resolution   120 \
    --use-gamma

# Seasonal + yearly, EPSG:3031 bbox, no GAMMA, IQR filter enabled
python run_pipeline.py \
    --itslive-dir  /data2/Cook/itslive_pot/v_vx_vy_v_error \
    --output-dir   /data2/Cook/output_velocity \
    --modes        seasonal yearly \
    --bbox-3031    -1800000 -1200000 -1500000 -900000 \
    --resolution   240 \
    --no-gamma \
    --iqr-filter

# All-data composite (single average), Landsat only
python run_pipeline.py \
    --itslive-dir  /data2/Cook/itslive_pot/v_vx_vy_v_error \
    --output-dir   /data2/Cook/output_velocity \
    --modes        all \
    --bbox-latlon  158.0 -68.5 163.5 -66.5 \
    --sensors      LC \
    --no-gamma
"""

import argparse
import sys
from pathlib import Path

# Allow running directly from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, TemporalMode, VxErrorMode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Glacier ice velocity weighted-average pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data directories ──────────────────────────────────────────────────────
    p.add_argument("--itslive-dir", dest="itslive_dirs", nargs="+",
                   default=["/data2/Cook/itslive_pot/v_vx_vy_v_error"],
                   metavar="DIR",
                   help="One or more directories containing ITS_LIVE TIF files")
    p.add_argument("--gamma-dir", dest="gamma_dirs", nargs="+",
                   default=["/data2/Cook/gamma_pot/v_vx_vy"],
                   metavar="DIR",
                   help="One or more directories containing GAMMA Sentinel-1 TIF files")
    p.add_argument("--output-dir",
                   default="/data2/Cook/output_velocity",
                   help="Root output directory")

    # ── BBox (mutually exclusive) ─────────────────────────────────────────────
    bbox_grp = p.add_mutually_exclusive_group(required=True)
    bbox_grp.add_argument("--bbox-latlon", nargs=4, type=float,
                           metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                           help="Bounding box in WGS-84 lon/lat")
    bbox_grp.add_argument("--bbox-3031", nargs=4, type=float,
                           metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                           help="Bounding box in EPSG:3031 metres (takes priority)")

    # ── Temporal modes ────────────────────────────────────────────────────────
    valid_modes = [m.value for m in TemporalMode]
    p.add_argument("--modes", nargs="+", choices=valid_modes,
                   default=["monthly"],
                   help="One or more temporal aggregation modes")

    # ── Grid ──────────────────────────────────────────────────────────────────
    p.add_argument("--resolution", type=float, default=120.0,
                   help="Output grid resolution in metres (EPSG:3031)")

    # ── Data sources ──────────────────────────────────────────────────────────
    # ITS_LIVE  (multi-sensor: S1 / S2 / Landsat)
    p.add_argument("--use-itslive", dest="use_itslive",
                   action="store_true", default=True,
                   help="Include ITS_LIVE data (default: on)")
    p.add_argument("--no-itslive", dest="use_itslive",
                   action="store_false",
                   help="Exclude ITS_LIVE data — use GAMMA only")
    # GAMMA POT  (Sentinel-1 offset tracking)
    p.add_argument("--use-gamma", dest="use_gamma",
                   action="store_true", default=True,
                   help="Include GAMMA Sentinel-1 data (default: on)")
    p.add_argument("--no-gamma", dest="use_gamma",
                   action="store_false",
                   help="Exclude GAMMA Sentinel-1 data — use ITS_LIVE only")

    p.add_argument("--sensors", nargs="+", choices=["S1", "S2", "LC"],
                   default=None,
                   help="Restrict ITS_LIVE to specific sensors")

    p.add_argument("--itslive-max-days", type=int, default=33,
                   help="Max temporal baseline (days) for ITS_LIVE pairs")
    p.add_argument("--gamma-max-days", type=int, default=None,
                   help="Max temporal baseline (days) for GAMMA pairs (None=all)")

    # ── Error estimation ──────────────────────────────────────────────────────
    err_choices = [m.value for m in VxErrorMode]
    p.add_argument("--vx-error-mode", choices=err_choices,
                   default="isotropic",
                   help="How to derive σVx/σVy from σV for ITS_LIVE data")
    p.add_argument("--gamma-error-mode", choices=err_choices,
                   default="local_std",
                   help="How to estimate σVx/σVy for GAMMA data (no error files)")
    p.add_argument("--local-std-window", type=int, default=5,
                   help="Window size (px) for local_std error estimation")

    # ── Spatial MAD filter (paper Stage 1) ───────────────────────────────────
    p.add_argument("--spatial-mad", dest="use_spatial_mad",
                   action="store_true", default=True,
                   help="Enable single-scene spatial MAD filter (paper Stage 1, "
                        "default: on)")
    p.add_argument("--no-spatial-mad", dest="use_spatial_mad",
                   action="store_false",
                   help="Disable spatial MAD filter")
    p.add_argument("--spatial-mad-window", type=int, default=5,
                   help="Odd window side in pixels for spatial MAD "
                        "(default 5 → 2×2 km at 400 m resolution)")
    p.add_argument("--spatial-mad-sigma", type=float, default=150.0,
                   metavar="M_PER_YR",
                   help="Window std upper limit in m/yr — paper value 150 m/yr "
                        "(default 150.0)")
    p.add_argument("--spatial-mad-min-valid", type=int, default=4,
                   help="Min finite pixels in window to compute stats "
                        "(default 4; avoids false positives at data edges)")
    p.add_argument("--spatial-mad-tile-rows", type=int, default=256,
                   help="Rows per processing stripe for spatial MAD "
                        "(default 256; reduce if memory-constrained)")
    p.add_argument("--spatial-mad-sources", nargs="+",
                   choices=["itslive", "gamma"],
                   default=["itslive", "gamma"],
                   help="Apply spatial MAD only to the selected data sources")

    # UMT spatial filter
    p.add_argument("--umt", dest="use_umt",
                   action="store_true", default=False,
                   help="Enable Universal Median Test (single-scene spatial filter)")
    p.add_argument("--no-umt", dest="use_umt",
                   action="store_false")
    p.add_argument("--umt-sources", nargs="+",
                   choices=["itslive", "gamma"],
                   default=["itslive", "gamma"],
                   help="Apply UMT only to the selected data sources")
    p.add_argument("--umt-window", type=int, default=5,
                   help="Odd window size (px) for UMT")
    p.add_argument("--umt-eps", type=float, default=0.1,
                   help="Minimum normalization level epsilon for UMT")
    p.add_argument("--umt-threshold", type=float, default=5.0,
                   help="Cull pixels where normalized UMT residual exceeds this")
    p.add_argument("--umt-min-segment", type=int, default=25,
                   help="Remove connected valid segments smaller than this after UMT")

    # ── IQR filter ────────────────────────────────────────────────────────────
    p.add_argument("--iqr-filter", dest="use_iqr_filter",
                   action="store_true", default=False,
                   help="Enable IQR-based outlier removal (Yang et al. 2026)")
    p.add_argument("--no-iqr-filter", dest="use_iqr_filter",
                   action="store_false")
    p.add_argument("--iqr-T", type=float, default=1.5,
                   help="IQR threshold multiplier T")
    p.add_argument("--iqr-cap", type=float, default=100.0,
                   help="Maximum IQR cap (m/yr)")

    # ── Coverage filter ───────────────────────────────────────────────────────
    p.add_argument("--min-coverage", type=float, default=0.0, metavar="FRAC",
                   help="Min valid-pixel fraction of bbox for a record to be used "
                        "(0-1). E.g. 0.05 = require >=5%% coverage. Default: 0.")

    # ── Temporal MAD filter ───────────────────────────────────────────────────
    p.add_argument("--temporal-mad", dest="use_temporal_mad",
                   action="store_true", default=True,
                   help="Enable per-pixel temporal MAD filter (default: on)")
    p.add_argument("--no-temporal-mad", dest="use_temporal_mad",
                   action="store_false",
                   help="Disable temporal MAD filter")
    p.add_argument("--temporal-mad-k", type=float, default=2.0,
                   help="MAD multiplier k: flag pixels where |Vx-median|>k*MAD "
                        "(default 2.0; use 3.0 for looser filter)")
    p.add_argument("--temporal-mad-min-n", type=int, default=3,
                   help="Min valid scenes per pixel to activate MAD filter "
                        "(default 3)")

    # ── Processing ────────────────────────────────────────────────────────────
    p.add_argument("--min-obs", type=int, default=1,
                   help="Minimum valid observations per pixel")
    p.add_argument("--error-threshold", type=float, default=None,
                   help="Cap σ (error) values above this (m/yr). Velocity is never clipped. Default: no cap")

    p.add_argument("--n-workers", type=int, default=1,
                   help="Groups processed in parallel (ProcessPoolExecutor). "
                        "With 80 CPUs try --n-workers 10 --io-threads 6")
    p.add_argument("--io-threads", type=int, default=4,
                   help="Threads for parallel record I/O within one group")

    # ── Date filter ───────────────────────────────────────────────────────────
    p.add_argument("--date-start", default=None,
                   help="Include only pairs with mid_date >= YYYY-MM-DD")
    p.add_argument("--date-end", default=None,
                   help="Include only pairs with mid_date <= YYYY-MM-DD")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--no-plots", dest="save_plots",
                   action="store_false", default=True,
                   help="Skip PNG overview generation")
    p.add_argument("--no-neff", dest="save_neff",
                   action="store_false", default=True,
                   help="Skip saving N_eff rasters")
    p.add_argument("--plot-dpi", type=int, default=150)
    p.add_argument("--vmax", type=float, default=None,
                   help="Colour-scale maximum (m/yr) for speed plots")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.spatial_mad_window < 3 or args.spatial_mad_window % 2 == 0:
        parser.error("--spatial-mad-window must be an odd integer >= 3")
    if args.spatial_mad_sigma <= 0:
        parser.error("--spatial-mad-sigma must be > 0")

    if not args.use_itslive and not args.use_gamma:
        parser.error("At least one data source must be enabled. "
                     "Remove --no-itslive or --no-gamma.")

    if args.umt_window < 3 or args.umt_window % 2 == 0:
        parser.error("--umt-window must be an odd integer >= 3")
    if args.umt_eps < 0:
        parser.error("--umt-eps must be >= 0")
    if args.umt_threshold <= 0:
        parser.error("--umt-threshold must be > 0")
    if args.umt_min_segment < 1:
        parser.error("--umt-min-segment must be >= 1")

    # Build PipelineConfig from CLI args
    cfg = PipelineConfig(
        itslive_dirs      = args.itslive_dirs,
        gamma_dirs        = args.gamma_dirs,
        output_dir        = args.output_dir,

        bbox_latlon       = tuple(args.bbox_latlon) if args.bbox_latlon else None,
        bbox_3031         = tuple(args.bbox_3031)   if args.bbox_3031   else None,

        temporal_modes    = [TemporalMode(m) for m in args.modes],
        target_resolution = args.resolution,

        use_spatial_mad       = args.use_spatial_mad,
        spatial_mad_window    = args.spatial_mad_window,
        spatial_mad_sigma_thr = args.spatial_mad_sigma,
        spatial_mad_min_valid = args.spatial_mad_min_valid,
        spatial_mad_tile_rows = args.spatial_mad_tile_rows,
        spatial_mad_sources   = args.spatial_mad_sources,

        use_itslive       = args.use_itslive,
        use_gamma         = args.use_gamma,
        itslive_sensors   = args.sensors,
        itslive_max_days  = args.itslive_max_days,
        gamma_max_days    = args.gamma_max_days,

        vx_error_mode     = VxErrorMode(args.vx_error_mode),
        gamma_error_mode  = VxErrorMode(args.gamma_error_mode),
        local_std_window  = args.local_std_window,
        use_umt           = args.use_umt,
        umt_sources       = args.umt_sources,
        umt_window_size   = args.umt_window,
        umt_epsilon       = args.umt_eps,
        umt_threshold     = args.umt_threshold,
        umt_min_segment_size = args.umt_min_segment,

        use_iqr_filter    = args.use_iqr_filter,
        iqr_threshold     = args.iqr_T,
        iqr_max_cap       = args.iqr_cap,

        min_valid_obs         = args.min_obs,
        min_coverage_frac     = args.min_coverage,
        use_temporal_mad      = args.use_temporal_mad,
        temporal_mad_k        = args.temporal_mad_k,
        temporal_mad_min_scenes = args.temporal_mad_min_n,
        error_threshold       = args.error_threshold,

        date_start        = args.date_start,
        date_end          = args.date_end,

        save_plots        = args.save_plots,
        save_neff         = args.save_neff,
        plot_dpi          = args.plot_dpi,
        plot_vmax         = args.vmax,

        n_workers         = args.n_workers,
        io_threads        = args.io_threads,
        log_level         = args.log_level,
    )

    # Run
    from pipeline import run_pipeline
    summaries = run_pipeline(cfg)

    if summaries:
        print(f"\n✓  Pipeline complete.  {len(summaries)} group(s) processed.")
        print(f"   Output root: {cfg.output_dir}")
    else:
        print("⚠  Pipeline produced no output.  Check logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
