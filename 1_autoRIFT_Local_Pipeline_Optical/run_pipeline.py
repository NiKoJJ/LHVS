#!/usr/bin/env python
"""
run_pipeline.py — autoRIFT_Optical_Local_Pipeline 主入口

【处理流程】
  Step 1  平台识别 & 场景日期解析
  Step 2  平台专属预处理滤波（L4/L5 FFT, L7 Wallis-fill, S2/L8/L9 透传）
  Step 3  GeogridOptical 配准元数据提取（coregister()）
  Step 4  构建 Geogrid 输入栅格（chip_size / search_range / vx0/vy0 / ssm）
  Step 5  运行 GeogridOptical → window_*.tif
  Step 6  autoRIFT 特征追踪 → 速度场 GeoTIFF / NetCDF

【目录结构】
  results/{pair_name}/
  ├── filtered/           预处理后影像（仅 L4/L5/L7）
  ├── geogrid_inputs/     chip_size / srx / vx0 等输入栅格
  ├── window_*.tif        Geogrid 输出
  ├── velocity_*.tif      最终速度场（GeoTIFF）
  ├── velocity_*.nc       最终速度场（NetCDF, 若 output_netcdf=true）
  └── offset_*.tif        原始像素偏移量（若 save_offsets=true）
"""

import sys as _sys
_argv_backup = _sys.argv[:]
_sys.argv = [_sys.argv[0]]
try:
    import isce3 as _isce3      # noqa: F401
except ImportError:
    pass
finally:
    _sys.argv = _argv_backup

import argparse
import logging
from pathlib import Path

import yaml

from src.utils import setup_logging, ensure_dir, get_epsg_from_file
from src.optical_processor import (
    detect_platform,
    get_scene_date,
    get_scene_name,
    apply_preprocessing,
    load_optical_metadata_pair,
)
from src.geogrid_builder_optical import (
    build_geogrid_inputs_optical,
    run_geogrid_optical,
)
from src.autorift_runner_optical import run_autorift_optical

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI & 配置
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="autoRIFT_Optical_Local_Pipeline — 本地光学影像速度场",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    p.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    p.add_argument(
        "--keep-intermediates", action="store_true", default=False,
        help="保留所有中间文件（默认：仅保留 velocity_*.tif / velocity_*.nc，其余删除）",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for key in ["reference_path", "secondary_path", "dem_file"]:
        val = cfg.get("input", {}).get(key, "")
        if not val or not Path(val).exists():
            raise FileNotFoundError(
                f"config.yaml → input.{key} = '{val}' 文件不存在，请检查路径。"
            )


def _parse_sps_files(cfg: dict) -> dict:
    sps_cfg = cfg.get("sps_params") or {}
    keys = [
        "dhdx_file", "dhdy_file",
        "vx0_file", "vy0_file",
        "search_range_x_file", "search_range_y_file",
        "chip_size_min_x_file", "chip_size_min_y_file",
        "chip_size_max_x_file", "chip_size_max_y_file",
    ]
    sps = {}
    for k in keys:
        v = sps_cfg.get(k) or None
        sps[k] = v

    provided = [k for k, v in sps.items() if v and Path(v).exists()]
    missing  = [k for k, v in sps.items() if v and not Path(v).exists()]
    if provided:
        log.info("SPS 参数文件 (%d 个): %s", len(provided), ", ".join(provided))
    if missing:
        log.warning("SPS 文件配置但不存在（已忽略）: %s", ", ".join(missing))
    if not provided:
        log.info("无 SPS 参数文件 → 使用均匀回退值")
    return sps


# ─────────────────────────────────────────────────────────────────────────────
# 中间文件清理
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_intermediates(work_dir: str) -> None:
    """删除工作目录中的所有中间文件，仅保留 velocity_*.tif 和 velocity_*.nc。

    删除内容：
      • 所有子目录（filtered/、geogrid_inputs/ 等）
      • window_*.tif  — Geogrid 输出窗口栅格
      • offset_*.tif  — 原始像素偏移量
      • *.yaml        — 每对自动生成的配置副本
      • 其余不符合保留规则的单文件

    保留内容：
      • velocity_*.tif
      • velocity_*.nc
    """
    import fnmatch
    import shutil

    keep_globs = ("velocity_*.tif", "velocity_*.nc")
    work_path  = Path(work_dir)
    n_files = n_dirs = 0

    for item in sorted(work_path.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
            n_dirs += 1
            log.debug("  删除子目录: %s", item.name)
        elif item.is_file():
            if not any(fnmatch.fnmatch(item.name, g) for g in keep_globs):
                item.unlink()
                n_files += 1
                log.debug("  删除文件: %s", item.name)

    kept = [p.name for p in work_path.iterdir() if p.is_file()]
    log.info(
        "中间文件清理完成：删除 %d 个文件 + %d 个子目录，保留: %s",
        n_files, n_dirs, kept if kept else "（无）",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("加载配置: %s", args.config)
    cfg = load_config(args.config)

    inp  = cfg["input"]
    proc = cfg["processing"]

    ref_path = inp["reference_path"]
    sec_path = inp["secondary_path"]
    dem_file = inp["dem_file"]
    ssm_file = inp.get("ssm_file") or None

    # ── Step 1: 平台 & 日期 ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 1: 平台识别与日期解析")

    platform = inp.get("platform") or detect_platform(ref_path)
    log.info("平台: %s", platform)

    date_ref = get_scene_date(ref_path)
    date_sec = get_scene_date(sec_path)
    log.info("参考景日期: %s  次景日期: %s", date_ref, date_sec)

    ref_name = inp.get("reference_scene") or get_scene_name(ref_path)
    sec_name = inp.get("secondary_scene") or get_scene_name(sec_path)

    pair_name = proc.get("pair_name") or f"{date_ref}_{date_sec}"
    log.info("像对名称: %s", pair_name)

    # ── 工作目录 ───────────────────────────────────────────────────────────────
    base_results_dir = proc.get("base_results_dir", "results")
    work_dir = str(Path(base_results_dir) / pair_name)
    ensure_dir(work_dir)
    log.info("工作目录: %s", work_dir)

    geogrid_dir = str(Path(work_dir) / "geogrid_inputs")
    window_dir  = work_dir

    # ── SPS 参数 ───────────────────────────────────────────────────────────────
    sps_files = _parse_sps_files(cfg)
    has_prior_v = bool(
        sps_files.get("vx0_file") and Path(sps_files["vx0_file"]).exists()
        and sps_files.get("vy0_file") and Path(sps_files["vy0_file"]).exists()
    )

    # ── Step 2: 预处理滤波 ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 2: 平台专属预处理滤波（%s）", platform)
    ref_proc, sec_proc, ref_zero_path, sec_zero_path = apply_preprocessing(
        ref_path, sec_path, platform, work_dir
    )

    # L7 零值掩膜（用于 autoRIFT zeroMask）
    zero_mask_ref = zero_mask_sec = None
    if ref_zero_path and sec_zero_path:
        import numpy as np
        from osgeo import gdal as _gdal
        _gdal.UseExceptions()
        _ds = _gdal.Open(ref_zero_path, _gdal.GA_ReadOnly)
        zero_mask_ref = _ds.GetRasterBand(1).ReadAsArray().astype(bool)
        _ds = None
        _ds = _gdal.Open(sec_zero_path, _gdal.GA_ReadOnly)
        zero_mask_sec = _ds.GetRasterBand(1).ReadAsArray().astype(bool)
        _ds = None

    # ── Step 3: 配准元数据提取 ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 3: GeogridOptical 配准元数据提取")
    meta_r, meta_s = load_optical_metadata_pair(
        ref_proc, sec_proc, ref_name=ref_name, sec_name=sec_name
    )

    # ── Step 4: Geogrid 输入栅格 ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 4: 构建 Geogrid 输入栅格")

    epsg = proc.get("epsg")
    if not epsg:
        epsg = get_epsg_from_file(dem_file)
    log.info("EPSG: %d", epsg)

    _csmin = proc.get("chip_size_min_m")
    _csmax = proc.get("chip_size_max_m")
    chip_size_min_m = float(_csmin) if _csmin is not None else None
    chip_size_max_m = float(_csmax) if _csmax is not None else None

    geogrid_files = build_geogrid_inputs_optical(
        dem_file=dem_file,
        ssm_file=ssm_file,
        output_dir=geogrid_dir,
        chip_size_min_m=chip_size_min_m,
        chip_size_max_m=chip_size_max_m,
        search_range_m_yr=float(proc["search_range_m_yr"]),
        sps_files=sps_files,
    )

    # ── Step 5: 运行 GeogridOptical ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 5: 运行 GeogridOptical")
    geogrid_run_info = run_geogrid_optical(
        meta_r=meta_r,
        meta_s=meta_s,
        epsg=epsg,
        geogrid_inputs=geogrid_files,
        working_dir=window_dir,
    )

    # ── Step 6: autoRIFT 特征追踪 ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 6: autoRIFT 特征追踪（光学模式）")

    vx0_for_error = geogrid_files["vx"] if has_prior_v else None
    vy0_for_error = geogrid_files["vy"] if has_prior_v else None

    results = run_autorift_optical(
        ref_path=ref_proc,
        sec_path=sec_proc,
        meta_r=meta_r,
        meta_s=meta_s,
        window_dir=window_dir,
        output_dir=work_dir,
        pair_name=pair_name,
        platform=platform,
        oversample_ratio=int(proc.get("oversample_ratio", 64)),
        n_threads=int(proc.get("n_threads", 4)),
        save_offsets=bool(proc.get("save_offsets", True)),
        velocity_max_m_yr=float(proc.get("velocity_max_m_yr", 20000.0)),
        vx0_tif=vx0_for_error,
        vy0_tif=vy0_for_error,
        save_netcdf=bool(proc.get("output_netcdf", False)),
        date_ref=date_ref,
        date_sec=date_sec,
        epsg=epsg,
        zero_mask_ref=zero_mask_ref,
        zero_mask_sec=zero_mask_sec,
        geogrid_run_info=geogrid_run_info,   # ← 新增这一行
    )

    # ── 完成 ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✓ 全部完成！")
    log.info("  平台:           %s", platform)
    log.info("  像对:           %s", pair_name)
    log.info("  速度场 GeoTIFF: %s", results["velocity_tif"])
    if "offset_tif" in results:
        log.info("  偏移量:         %s", results["offset_tif"])
    if "netcdf" in results:
        log.info("  NetCDF CF-1.8:  %s", results["netcdf"])
    log.info("  Band: Vx(m/yr) / Vy(m/yr) / |V|(m/yr)  NoData=%g", -32767.0)
    log.info(
        "  误差: Vx_err=%.2f  Vy_err=%.2f  V_err=%.2f  stable_count=%d  [m/yr]",
        results.get("vx_error",  float("nan")),
        results.get("vy_error",  float("nan")),
        results.get("v_error",   float("nan")),
        results.get("stable_count", 0),
    )
    if not has_prior_v:
        log.info("  （误差为稳定地表标准差估计；提供 SPS vx0/vy0 可获更准确误差）")

    # ── 中间文件清理 ──────────────────────────────────────────────────────────
    if not args.keep_intermediates:
        log.info("=" * 60)
        log.info("清理中间文件（保留 velocity_*.tif / velocity_*.nc）…")
        _cleanup_intermediates(work_dir)
    else:
        log.info("--keep-intermediates 已启用，跳过中间文件清理")


if __name__ == "__main__":
    main()
