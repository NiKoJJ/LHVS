"""
pipeline.py  —  two-level parallel orchestrator

  io_threads  (default 4)  ThreadPoolExecutor per group  →  parallel file I/O
  n_workers   (default 1)  ProcessPoolExecutor           →  parallel groups

Filtering order within each group:
  0a. Spatial MAD filter    (optional, per-record, single-scene spatial filter)
      → paper Stage 1: 5×5 window median/std/MAD, removes outlier pixels
  0b. Universal Median Test (optional, per-record spatial filter)
  1.  Coverage filter       (per-record, at read time)
  2.  Temporal MAD          (per-pixel stack)
  3.  IQR filter            (per-pixel stack)
  4.  Weighted average
"""
from __future__ import annotations
import json, logging, multiprocessing, time, warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

from config import PipelineConfig, TemporalMode
from data_sources import load_all_records, records_to_dataframe
from spatial import resolve_bbox_3031, build_target_grid, read_triplet, read_and_reproject
from error_and_outlier import (estimate_vx_vy_errors, apply_iqr_filter,
                               apply_temporal_mad_filter,
                               apply_universal_median_test,
                               apply_spatial_mad_filter)
from temporal import assign_groups, write_group_log
from weighted_avg import process_stack
from io_utils import save_group_outputs
from visualize import save_group_figure, save_timeseries_summary

log = logging.getLogger(__name__)


# ── per-record reader (thread worker) ────────────────────────────────────────

def _read_one(args):
    """Read one record; returns (index, data_dict | None)."""
    idx, rec, cfg, tf, W, H, crs = args
    arrays = read_triplet(rec, tf, W, H, crs, no_data_value=cfg.no_data_value)
    if arrays is None:
        return idx, None

    vx, vy, v, ve = arrays["vx"], arrays["vy"], arrays["v"], arrays["v_error"]
    if vx is None or vy is None or v is None:
        return idx, None

    mode = cfg.vx_error_mode.value if rec.source == "itslive" \
           else cfg.gamma_error_mode.value

    # ── Error: use direct files when available, otherwise estimate ────────────
    vx_e = vy_e = None

    if rec.vx_error_path and rec.vy_error_path:
        vx_e = read_and_reproject(rec.vx_error_path, tf, W, H, crs,
                                   no_data_value=cfg.no_data_value)
        vy_e = read_and_reproject(rec.vy_error_path, tf, W, H, crs,
                                   no_data_value=cfg.no_data_value)
        if vx_e is not None: vx_e[vx_e <= 0] = np.nan
        if vy_e is not None: vy_e[vy_e <= 0] = np.nan

    if vx_e is None or vy_e is None:
        vx_e, vy_e = estimate_vx_vy_errors(
            vx, vy, ve, mode=mode,
            window_size=cfg.local_std_window, min_valid=cfg.local_std_min_valid)

    v_e = ve.copy() if ve is not None else (
          np.sqrt(vx_e**2 + vy_e**2).astype(np.float32)
          if vx_e is not None else np.full_like(vx, np.nan))

    if cfg.use_spatial_mad and rec.source in set(cfg.spatial_mad_sources):
        vx, vy, v, vx_e, vy_e, v_e, smad_stats = apply_spatial_mad_filter(
            vx, vy, v=v, vx_err=vx_e, vy_err=vy_e, v_err=v_e,
            window_size=cfg.spatial_mad_window,
            sigma_threshold=cfg.spatial_mad_sigma_thr,
            min_valid=cfg.spatial_mad_min_valid,
            tile_rows=cfg.spatial_mad_tile_rows,
        )
        if smad_stats["n_removed"] > 0:
            log.debug(
                "  SpatialMAD %s: removed %d px (%.1f%%) "
                "[crit_a=%d  crit_b_only=%d]",
                rec.label,
                smad_stats["n_removed"], smad_stats["removal_pct"],
                smad_stats["n_crit_a"],  smad_stats["n_crit_b_only"],
            )

    if cfg.use_umt and rec.source in set(cfg.umt_sources):
        vx, vy, v, vx_e, vy_e, v_e, umt_stats = apply_universal_median_test(
            vx, vy, v=v, vx_err=vx_e, vy_err=vy_e, v_err=v_e,
            window_size=cfg.umt_window_size,
            epsilon=cfg.umt_epsilon,
            threshold=cfg.umt_threshold,
            min_segment_size=cfg.umt_min_segment_size,
        )
        if (umt_stats["n_outlier_pixels"] > 0
                or umt_stats["n_small_segment_pixels"] > 0):
            log.debug(
                "  UMT %s: removed %d outlier px, %d small-segment px",
                rec.label,
                umt_stats["n_outlier_pixels"],
                umt_stats["n_small_segment_pixels"],
            )

    if cfg.error_threshold is not None:
        thr = cfg.error_threshold
        vx_e[vx_e > thr] = np.nan
        vy_e[vy_e > thr] = np.nan
        v_e [v_e  > thr] = np.nan

    # Coverage fraction after any single-scene spatial culling.
    coverage_frac = float(np.sum(np.isfinite(vx)) / vx.size)

    return idx, dict(vx=vx, vy=vy, v=v, vx_e=vx_e, vy_e=vy_e, v_e=v_e,
                     coverage_frac=coverage_frac)


# ── single group ─────────────────────────────────────────────────────────────

def run_group(group_name, records, cfg, transform, width, height,
              target_crs, mode_str, out_mode_dir):
    N = len(records)
    if N < cfg.min_valid_obs:
        return None

    n_thr = min(cfg.io_threads, N)
    args  = [(i, r, cfg, transform, width, height, target_crs)
             for i, r in enumerate(records)]

    if n_thr > 1:
        with ThreadPoolExecutor(max_workers=n_thr) as pool:
            results = dict(pool.map(_read_one, args))
    else:
        results = dict(_read_one(a) for a in args)

    # ── Coverage filter + collect stacks ──────────────────────────────────────
    vx_l, vy_l, v_l, vxe_l, vye_l, ve_l = [], [], [], [], [], []
    record_stats: list = []

    for i in range(N):
        d   = results.get(i)
        rec = records[i]
        if d is None:
            record_stats.append(dict(record=rec, coverage_frac=0.0,
                                     participated=False, skip_reason="no_overlap"))
            continue
        cov = d.get("coverage_frac", 0.0)
        if cov < cfg.min_coverage_frac:
            record_stats.append(dict(record=rec, coverage_frac=cov,
                                     participated=False,
                                     skip_reason=f"coverage {cov*100:.1f}% "
                                                 f"< threshold {cfg.min_coverage_frac*100:.1f}%"))
            continue
        record_stats.append(dict(record=rec, coverage_frac=cov,
                                 participated=True, skip_reason=None))
        vx_l.append(d["vx"]); vy_l.append(d["vy"]); v_l.append(d["v"])
        vxe_l.append(d["vx_e"]); vye_l.append(d["vy_e"]); ve_l.append(d["v_e"])

    read_ok   = len(vx_l)
    n_excluded = sum(1 for s in record_stats if not s["participated"])
    if n_excluded:
        log.info("  Coverage filter excluded %d / %d records (threshold=%.0f%%)",
                 n_excluded, N, cfg.min_coverage_frac * 100)
    if read_ok == 0:
        log.warning("  SKIP %s – no bbox overlap", group_name)
        return None

    def _s(lst): return np.stack(lst, axis=0).astype(np.float32)
    vx_stk  = _s(vx_l);  vy_stk  = _s(vy_l);  v_stk  = _s(v_l)
    vxe_stk = _s(vxe_l); vye_stk = _s(vye_l); ve_stk = _s(ve_l)

    # ── [NEW] Temporal MAD filter ─────────────────────────────────────────────
    # Per-pixel: flag scenes where |Vx - median_Vx| > k * MAD_Vx (same for Vy)
    # Only meaningful with >= temporal_mad_min_scenes scenes.
    if cfg.use_temporal_mad and read_ok >= cfg.temporal_mad_min_scenes:
        vx_stk, vy_stk, v_stk, vxe_stk, vye_stk = apply_temporal_mad_filter(
            vx_stk, vy_stk, v_stk, vxe_stk, vye_stk,
            k=cfg.temporal_mad_k,
            min_scenes=cfg.temporal_mad_min_scenes)
        n_surviving = int(np.sum(np.any(np.isfinite(vx_stk), axis=(1, 2))))
        log.info("  Temporal MAD (k=%.1f): %d / %d scenes have surviving pixels",
                 cfg.temporal_mad_k, n_surviving, read_ok)

    # ── IQR filter ────────────────────────────────────────────────────────────
    if cfg.use_iqr_filter and read_ok > 1:
        vx_stk, vy_stk, v_stk, vxe_stk, vye_stk = apply_iqr_filter(
            vx_stk, vy_stk, v_stk, vxe_stk, vye_stk,
            T=cfg.iqr_threshold, iqr_max_cap=cfg.iqr_max_cap)

    result = process_stack(vx_stk, vy_stk, v_stk, vxe_stk, vye_stk, ve_stk,
                           min_obs=cfg.min_valid_obs)
    result["n_records"] = read_ok

    out_sub = Path(out_mode_dir) / group_name
    save_group_outputs(group_name, result, out_sub, transform, target_crs,
                       save_neff=cfg.save_neff)
    if cfg.save_plots:
        save_group_figure(group_name, result, out_sub,
                          target_transform=transform,
                          target_crs_str=cfg.target_crs,
                          dpi=cfg.plot_dpi, vmax=cfg.plot_vmax)

    v_main    = result.get("v_synth", result.get("v"))
    mid_dates = [r.mid_date for r in records]
    mean_date = min(mid_dates) + (max(mid_dates) - min(mid_dates)) / 2
    return dict(
        mode=mode_str, group=group_name,
        n_records=read_ok, n_total=N,
        mean_date=mean_date,
        mean_v     = float(np.nanmean(v_main))
                     if v_main is not None else None,
        mean_vx    = float(np.nanmean(result["vx"]))
                     if result.get("vx") is not None else None,
        mean_vy    = float(np.nanmean(result["vy"]))
                     if result.get("vy") is not None else None,
        mean_v_err = float(np.nanmean(result.get("v_synth_err", result.get("v_err"))))
                     if result.get("v_synth_err") is not None else None,
        coverage_pct = float(np.sum(np.isfinite(v_main)) / v_main.size * 100)
                       if v_main is not None else 0.0,
        out_dir      = str(out_sub),
        record_stats = record_stats,
    )


# ── process-pool worker (must be top-level for pickle) ───────────────────────

def _group_worker(kw):
    logging.basicConfig(
        level=kw["log_level"],
        format="%(asctime)s [%(levelname)s][%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    try:
        return run_group(
            kw["group_name"], kw["records"], kw["cfg"],
            kw["transform"], kw["width"], kw["height"], kw["target_crs"],
            kw["mode_str"], kw["out_mode_dir"])
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Group %s failed: %s", kw["group_name"], exc, exc_info=True)
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> List[dict]:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    t0 = time.time()
    log.info("=" * 65)
    log.info("  Glacier Velocity Pipeline")
    log.info("  modes=%s  n_workers=%d  io_threads=%d",
             [m.value for m in cfg.temporal_modes], cfg.n_workers, cfg.io_threads)
    log.info("  spatial_MAD=%s (ws=%d, σ_thr=%.0f)  temporal_MAD=%s (k=%.1f, min_n=%d)  IQR=%s (T=%.1f)",
             cfg.use_spatial_mad, cfg.spatial_mad_window, cfg.spatial_mad_sigma_thr,
             cfg.use_temporal_mad, cfg.temporal_mad_k, cfg.temporal_mad_min_scenes,
             cfg.use_iqr_filter, cfg.iqr_threshold)
    log.info("=" * 65)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_all_records(cfg)
    if not records:
        log.error("No records – aborting."); return []

    records_to_dataframe(records).to_csv(out_root / "scene_inventory.csv", index=False)

    bbox_3031 = resolve_bbox_3031(cfg)
    transform, W, H, crs = build_target_grid(bbox_3031, cfg.target_resolution,
                                              cfg.target_crs)
    with open(out_root / "grid_info.json", "w") as f:
        json.dump({"bbox_3031": list(bbox_3031),
                   "bbox_latlon": list(cfg.bbox_latlon) if cfg.bbox_latlon else None,
                   "crs": cfg.target_crs, "resolution": cfg.target_resolution,
                   "width": W, "height": H, "transform": list(transform)}, f, indent=2)

    all_summaries: List[dict] = []

    for mode in cfg.temporal_modes:
        mode_str     = mode.value
        out_mode_dir = out_root / mode_str
        out_mode_dir.mkdir(exist_ok=True)
        log.info("\n── Mode: %s", mode_str)

        groups = assign_groups(records, mode)
        if not groups: continue

        gnames = list(groups.keys())
        nG     = len(gnames)
        mode_summ:       List[dict] = []
        all_group_stats: dict       = {}   # {group_name: [record_stat_dicts]}

        if cfg.n_workers > 1 and nG > 1:
            log.info("Parallel: %d groups × %d workers", nG, cfg.n_workers)
            work = [dict(
                group_name=g, records=groups[g], cfg=cfg,
                transform=transform, width=W, height=H, target_crs=crs,
                mode_str=mode_str, out_mode_dir=str(out_mode_dir),
                log_level=getattr(logging, cfg.log_level.upper(), logging.INFO))
                for g in gnames]
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=cfg.n_workers, mp_context=ctx) as pool:
                futs = {pool.submit(_group_worker, w): w["group_name"] for w in work}
                done = 0
                for fut in as_completed(futs):
                    g = futs[fut]; done += 1
                    try:
                        s = fut.result()
                    except Exception as e:
                        log.error("[%d/%d] %s FAILED: %s", done, nG, g, e); s = None
                    if s:
                        s["mode"] = mode_str; mode_summ.append(s)
                        all_group_stats[g] = s.pop("record_stats", [])
                        log.info("[%d/%d] ✓ %s  files=%d/%d  cov=%.1f%%  V=%.1f",
                                 done, nG, g,
                                 s["n_records"], s["n_total"],
                                 s["coverage_pct"], s["mean_v"] or 0)
        else:
            for i, g in enumerate(gnames, 1):
                log.info("[%d/%d] %s", i, nG, g)
                tg = time.time()
                try:
                    s = run_group(g, groups[g], cfg, transform, W, H, crs,
                                  mode_str, out_mode_dir)
                except Exception as e:
                    log.error("  ERROR: %s", e, exc_info=True); s = None
                if s:
                    s["mode"] = mode_str; mode_summ.append(s)
                    all_group_stats[g] = s.pop("record_stats", [])
                    log.info("  ✓ %.1fs  files=%d/%d  cov=%.1f%%  V=%.1f",
                             time.time()-tg,
                             s["n_records"], s["n_total"],
                             s["coverage_pct"], s["mean_v"] or 0)

        if mode_summ:
            pd.DataFrame(mode_summ).to_csv(
                out_mode_dir / f"summary_{mode_str}.csv", index=False)
            if cfg.save_plots:
                save_timeseries_summary(mode_summ, out_mode_dir, dpi=cfg.plot_dpi)

        write_group_log(str(out_mode_dir / f"group_log_{mode_str}.txt"),
                        mode, groups, cfg,
                        group_stats=all_group_stats if all_group_stats else None)
        all_summaries += mode_summ

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(out_root / "master_summary.csv", index=False)

    elapsed = time.time() - t0
    log.info("DONE  %.1fs (%.1f min)  →  %s", elapsed, elapsed/60, out_root)
    return all_summaries
