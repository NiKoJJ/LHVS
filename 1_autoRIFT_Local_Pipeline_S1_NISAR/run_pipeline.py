#!/usr/bin/env python
"""
run_pipeline.py — autoRIFT_SAR_Local_Pipeline 主入口
【v4 新增】
  ✦ NISAR 支持：RSLC（斜距，ISCE3 协同配准）& GSLC（地理编码）
  ✦ 传感器自动检测：.zip → S1；.h5 → NISAR
  ✦ 接口向后兼容：S1 流程完全不变

【目录结构】
results/{pair_name}/
├── product/          : 参考景 Burst 产品（S1）
├── product_sec/      : 次景 Burst 产品（S1）
├── scratch/          : 临时几何文件（S1 COMPASS / NISAR ISCE3）
├── nisar_ref_cropped.tif / nisar_sec_cropped.tif  (NISAR GSLC)
├── reference.tif     : 最终参考幅度图
├── secondary.tif     : 最终次景幅度图
├── geogrid_inputs/   : Geogrid 输入栅格
├── window_*.tif      : Geogrid 输出
├── velocity_*.tif    : 最终速度场（GeoTIFF）
├── velocity_*.nc     : 最终速度场（NetCDF，可选）
└── offset_*.tif      : 像素偏移量（可选）
"""
# ═══════════════════════════════════════════════════════════════════════════════
#  预初始化 isce3 / s1reader（必须在 argparse 之前）
# ═══════════════════════════════════════════════════════════════════════════════
import sys as _sys
_argv_backup = _sys.argv[:]
_sys.argv = [_sys.argv[0]]
try:
    import isce3 as _isce3       # noqa: F401
    import s1reader as _s1r      # noqa: F401
except ImportError:
    pass
finally:
    _sys.argv = _argv_backup
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import logging
from pathlib import Path

import yaml

from src.utils import setup_logging, ensure_dir, get_epsg_from_file

log = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="autoRIFT_SAR_Local_Pipeline v4 — S1 & NISAR 速度场",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default="config.yaml", help="配置文件路径")
    p.add_argument("--list-bursts", action="store_true",
                   help="列出 S1 参考景所有可用 burst 后退出")
    p.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    p.add_argument("--keep-intermediates", action="store_true", default=False,
                   help="保留所有中间文件（默认：仅保留 velocity_*.tif / velocity_*.nc）")
    return p.parse_args()


# ── 配置加载 & 校验 ────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    inp = cfg.get("input", {})
    proc = cfg.get("processing", {})
    sensor = _detect_sensor_from_config(cfg)

    for key in ("output_resolution_m", "chip_size_min_m", "chip_size_max_m", "search_range_m_yr"):
        value = proc.get(key)
        if value is None:
            continue
        try:
            if float(value) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"processing.{key} 必须为正数，或在允许时设为 null") from exc

    if sensor == 'S1':
        for key in ["reference_zip", "secondary_zip", "dem_file"]:
            val = inp.get(key, "")
            if not val or not Path(val).exists():
                raise FileNotFoundError(
                    f"config.yaml → input.{key} = '{val}' 文件不存在，请检查路径。"
                )
    elif sensor == 'NISAR':
        for key in ["reference_h5", "secondary_h5", "dem_file"]:
            val = inp.get(key, "")
            if not val or not Path(val).exists():
                raise FileNotFoundError(
                    f"config.yaml → input.{key} = '{val}' 文件不存在，请检查路径。"
                )
        # Check the HDF5 container before starting the expensive ISCE3 steps.
        # A partially copied NISAR product otherwise fails much later in
        # geo2rdr with an opaque "truncated file" traceback.
        key = "reference_h5"
        try:
            import h5py
            for key in ("reference_h5", "secondary_h5"):
                with h5py.File(inp[key], "r", libver="latest", swmr=True):
                    pass
        except Exception as exc:
            raise ValueError(
                "NISAR HDF5 文件无法读取（可能是复制/下载未完成）: "
                f"{inp.get(key, '')}. 请重新复制或下载该文件。原始错误: {exc}"
            ) from exc


# ── 传感器检测 ─────────────────────────────────────────────────────────────────
def _detect_sensor_from_config(cfg: dict) -> str:
    """从 config.yaml input 节判断传感器类型。

    优先根据字段名称：
      reference_h5  → NISAR
      reference_zip → S1
    """
    inp = cfg.get("input", {})
    if inp.get("reference_h5"):
        return 'NISAR'
    if inp.get("reference_zip"):
        return 'S1'
    # 最后按文件扩展名猜测
    ref = inp.get("reference_zip", "") or inp.get("reference_h5", "")
    if ref.endswith('.h5') or ref.endswith('.hdf5'):
        return 'NISAR'
    return 'S1'


# ── 日期解析 ───────────────────────────────────────────────────────────────────
def _date_from_file(path: str, sensor: str) -> str:
    """从文件名提取日期 YYYYMMDD。支持 S1 zip 和 NISAR h5 命名约定。"""
    stem = Path(path).stem
    if sensor == 'S1':
        # S1A_IW_SLC__1SSH_20220906T100810_...
        try:
            return stem.split("_")[5][:8]
        except (IndexError, ValueError):
            return stem[:8]
    elif sensor == 'NISAR':
        # NISAR_L1_PR_RSLC_001_001_A_..._20240101T120000_...
        # 尝试找 T 字段中的日期
        for part in stem.split("_"):
            if len(part) == 15 and 'T' in part:
                return part[:8]
        # 回退：找第一个 8 位纯数字
        for part in stem.split("_"):
            if len(part) == 8 and part.isdigit():
                return part
        return stem[:8]
    return stem[:8]


# ── SPS 参数解析（S1 流程保持不变）──────────────────────────────────────────────
def _parse_sps_files(cfg: dict) -> dict:
    sps_cfg = cfg.get("sps_params") or {}
    keys = [
        "dhdx_file", "dhdy_file",
        "vx0_file",  "vy0_file",
        "search_range_x_file", "search_range_y_file",
        "chip_size_min_x_file", "chip_size_min_y_file",
        "chip_size_max_x_file", "chip_size_max_y_file",
    ]
    sps = {k: sps_cfg.get(k) or None for k in keys}

    provided = [k for k, v in sps.items() if v and Path(v).exists()]
    missing  = [k for k, v in sps.items() if v and not Path(v).exists()]
    if provided:
        log.info("SPS 参数文件 (%d 个): %s", len(provided), ", ".join(provided))
    if missing:
        log.warning("SPS 文件路径已配置但文件不存在（已忽略）: %s", ", ".join(missing))
    if not provided:
        log.info("未检测到有效 SPS 参数文件 → 将使用均匀回退值")
    return sps


# ── 中间文件清理 ───────────────────────────────────────────────────────────────
def _cleanup_intermediates(work_dir: str) -> None:
    """删除中间文件，仅保留 velocity_*.tif 和 velocity_*.nc。"""
    import fnmatch
    import shutil

    keep_globs = ("velocity_*.tif", "velocity_*.nc", "*.yaml")
    work_path  = Path(work_dir)
    n_files = n_dirs = 0

    for item in sorted(work_path.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
            n_dirs += 1
        elif item.is_file():
            if not any(fnmatch.fnmatch(item.name, g) for g in keep_globs):
                item.unlink()
                n_files += 1

    kept = [p.name for p in work_path.iterdir() if p.is_file()]
    log.info(
        "中间文件清理完成：删除 %d 个文件 + %d 个子目录，保留: %s",
        n_files, n_dirs, kept if kept else "（无）",
    )


# ══════════════════════════════════════════════════════════════════════════════
# S1 流程（原 main 主体，保持不变）
# ══════════════════════════════════════════════════════════════════════════════
def _run_s1(cfg: dict, work_dir: str, args) -> dict:
    """执行 Sentinel-1 处理流程（与 v3 完全一致）。"""
    from src.s1_processor import (
        get_common_burst_ids,
        get_orbit_file,
        process_all_bursts_compass,
        merge_burst_amplitudes,
        load_slc_metadata_pair,
    )
    from src.geogrid_builder import build_geogrid_inputs, prepare_geogrid_dem, run_geogrid_sar
    from src.autorift_runner import run_autorift

    inp  = cfg["input"]
    orb  = cfg["orbit"]
    proc = cfg["processing"]

    ref_zip  = inp["reference_zip"]
    sec_zip  = inp["secondary_zip"]
    dem_file = inp["dem_file"]
    ssm_file = inp.get("ssm_file") or None

    date_ref  = _date_from_file(ref_zip, 'S1')
    date_sec  = _date_from_file(sec_zip, 'S1')
    pair_name = proc.get("pair_name") or f"{date_ref}_{date_sec}"

    amp_ref     = str(Path(work_dir) / "reference.tif")
    amp_sec     = str(Path(work_dir) / "secondary.tif")
    window_dir  = work_dir
    geogrid_dir = str(Path(work_dir) / "geogrid_inputs")

    product_dir_ref = str(Path(work_dir) / "product")
    product_dir_sec = str(Path(work_dir) / "product_sec")

    # ── --list-bursts ────────────────────────────────────────────────────────
    if args.list_bursts:
        try:
            orbit_ref = get_orbit_file(ref_zip, orb["orbit_dir"], allow_download=False)
        except FileNotFoundError:
            orbit_ref = None
        from src.s1_processor import get_all_bursts
        records = get_all_bursts(ref_zip, orbit_ref)
        print(f"\n参考景可用 Burst（共 {len(records)} 个）：{Path(ref_zip).name}\n")
        print(f"{'swath':>6}  {'burst_idx':>9}  {'ISCE3 ID':>24}  {'sensing_start':>22}")
        print("-" * 70)
        for r in records:
            print(
                f"  IW{r['swath']}  {r['burst_index']:>9}  "
                f"{r['isce3_id']:>24}  {r['sensing_start'][:22]}"
            )
        print()
        _sys.exit(0)

    # ── SPS 参数 ─────────────────────────────────────────────────────────────
    sps_files = _parse_sps_files(cfg)
    has_prior_v = bool(
        sps_files.get("vx0_file") and Path(sps_files["vx0_file"]).exists()
        and sps_files.get("vy0_file") and Path(sps_files["vy0_file"]).exists()
    )

    # Step 1: 轨道文件
    log.info("Step 1: 获取轨道文件")
    orbit_ref = get_orbit_file(ref_zip, orb["orbit_dir"], allow_download=orb["allow_download"])
    orbit_sec = get_orbit_file(sec_zip, orb["orbit_dir"], allow_download=orb["allow_download"])

    # Step 2: Burst 交集
    log.info("Step 2: 计算共视 Burst 交集")
    common_ids, swaths_used = get_common_burst_ids(ref_zip, sec_zip, orbit_ref, orbit_sec)
    log.info("共视 Burst: %d | 使用 Swath: %s", len(common_ids), swaths_used)

    # Step 3 + 4: COMPASS + 合并幅度图
    if Path(amp_ref).exists() and Path(amp_sec).exists():
        log.info("幅度图已存在，跳过 COMPASS 与合并步骤（断点续算）。")
    else:
        log.info("Step 3: COMPASS CSLC（%d burst × 2 景）", len(common_ids))
        process_all_bursts_compass(
            ref_zip=ref_zip, sec_zip=sec_zip,
            orbit_ref=orbit_ref, orbit_sec=orbit_sec,
            burst_ids_ref=common_ids, burst_ids_sec=common_ids,
            dem_file=dem_file,
            product_dir_ref=product_dir_ref, product_dir_sec=product_dir_sec,
            work_dir=work_dir,
        )
        log.info("Step 4: 合并 burst 幅度图")
        merge_burst_amplitudes(
            burst_ids=common_ids, ref_zip=ref_zip, orbit_ref=orbit_ref,
            product_dir_ref=product_dir_ref, product_dir_sec=product_dir_sec,
            amp_ref_path=amp_ref, amp_sec_path=amp_sec,
        )

    # Step 5: Geogrid 输入
    log.info("Step 5: 构建 Geogrid 输入栅格")
    output_resolution_m = proc.get("output_resolution_m")
    geogrid_dem = prepare_geogrid_dem(
        dem_file=dem_file,
        output_dir=geogrid_dir,
        output_resolution_m=float(output_resolution_m) if output_resolution_m is not None else None,
        reuse_existing=bool(proc.get("reuse_geogrid_dem", True)),
    )
    epsg = proc.get("epsg") or get_epsg_from_file(geogrid_dem)
    log.info("EPSG: %d", epsg)

    _csmin = proc.get("chip_size_min_m")
    _csmax = proc.get("chip_size_max_m")

    geogrid_files = build_geogrid_inputs(
        dem_file=geogrid_dem, ssm_file=ssm_file, output_dir=geogrid_dir,
        chip_size_min_m=float(_csmin) if _csmin is not None else None,
        chip_size_max_m=float(_csmax) if _csmax is not None else None,
        search_range_m_yr=float(proc["search_range_m_yr"]),
        sps_files=sps_files,
    )

    # Step 6: Geogrid
    log.info("Step 6: runGeogrid（Swath %s）", swaths_used)
    meta_r, meta_s = load_slc_metadata_pair(
        ref_zip=ref_zip, sec_zip=sec_zip,
        orbit_ref=orbit_ref, orbit_sec=orbit_sec,
        swaths=swaths_used,
    )
    geogrid_run_info = run_geogrid_sar(
        meta_r=meta_r, meta_s=meta_s,
        epsg=epsg, geogrid_inputs=geogrid_files,
        working_dir=window_dir,
        optical_flag=0,   # S1 始终为斜距几何
    )

    # Step 7: autoRIFT
    log.info("Step 7: autoRIFT 特征追踪")
    vx0_for_error = geogrid_files["vx"] if has_prior_v else None
    vy0_for_error = geogrid_files["vy"] if has_prior_v else None

    results = run_autorift(
        ref_tif=amp_ref, sec_tif=amp_sec,
        window_dir=window_dir, output_dir=work_dir,
        pair_name=pair_name,
        filter_type=proc.get("filter_type", "WAL"),
        filter_width=int(proc.get("filter_width", 21)),
        oversample_ratio=int(proc.get("oversample_ratio", 64)),
        n_threads=int(proc.get("n_threads", 4)),
        save_offsets=bool(proc.get("save_offsets", True)),
        velocity_max_m_yr=float(proc.get("velocity_max_m_yr", 20000.0)),
        vx0_tif=vx0_for_error, vy0_tif=vy0_for_error,
        save_netcdf=bool(proc.get("output_netcdf", False)),
        date_ref=date_ref, date_sec=date_sec, epsg=epsg,
        # Geogrid 的实际像元/格网参数决定 ChipSize0X，S1 与 NISAR 都必须传入。
        geogrid_run_info=geogrid_run_info,
    )

    if not has_prior_v:
        log.info("（误差为稳定地表标准差估计；提供 SPS vx0/vy0 可获得更准确的误差）")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# NISAR 流程（v4 新增）
# ══════════════════════════════════════════════════════════════════════════════
def _run_nisar(cfg: dict, work_dir: str, args) -> dict:
    """执行 NISAR RSLC/GSLC 处理流程。"""
    from src.nisar_processor import process_nisar
    from src.geogrid_builder import build_geogrid_inputs, prepare_geogrid_dem, run_geogrid_sar
    from src.autorift_runner import run_autorift

    inp  = cfg["input"]
    proc = cfg["processing"]

    ref_h5   = inp["reference_h5"]
    sec_h5   = inp["secondary_h5"]
    dem_file = inp["dem_file"]
    ssm_file = inp.get("ssm_file") or None

    nisar_cfg    = cfg.get("nisar", {})
    frequency    = nisar_cfg.get("frequency",    "A")

    date_ref  = _date_from_file(ref_h5, 'NISAR')
    date_sec  = _date_from_file(sec_h5, 'NISAR')
    pair_name = proc.get("pair_name") or f"{date_ref}_{date_sec}"

    amp_ref     = str(Path(work_dir) / "reference.tif")
    amp_sec     = str(Path(work_dir) / "secondary.tif")
    window_dir  = work_dir
    geogrid_dir = str(Path(work_dir) / "geogrid_inputs")

    # ── SPS 参数 ─────────────────────────────────────────────────────────────
    sps_files = _parse_sps_files(cfg)
    has_prior_v = bool(
        sps_files.get("vx0_file") and Path(sps_files["vx0_file"]).exists()
        and sps_files.get("vy0_file") and Path(sps_files["vy0_file"]).exists()
    )

    # Step 3: NISAR 预处理（检测类型、协同配准/裁剪、幅度提取、元数据）
    log.info("=" * 60)
    log.info("Step 3: NISAR 预处理（nisar_processor.process_nisar）")
    meta_r, meta_s, optical_flag = process_nisar(
        ref_h5=ref_h5, sec_h5=sec_h5,
        dem_file=dem_file,
        work_dir=work_dir,
        amp_ref_path=amp_ref,
        amp_sec_path=amp_sec,
        frequency=frequency,
    )
    log.info("optical_flag = %d（%s）", optical_flag,
             "斜距几何 RSLC" if optical_flag == 0 else "地理编码 GSLC")

    # Step 5: Geogrid 输入
    log.info("=" * 60)
    log.info("Step 5: 构建 Geogrid 输入栅格")
    output_resolution_m = proc.get("output_resolution_m")
    geogrid_dem = prepare_geogrid_dem(
        dem_file=dem_file,
        output_dir=geogrid_dir,
        output_resolution_m=float(output_resolution_m) if output_resolution_m is not None else None,
        reuse_existing=bool(proc.get("reuse_geogrid_dem", True)),
    )
    epsg = proc.get("epsg")
    if not epsg:
        # GSLC 幅度图含地理坐标，可直接读取；RSLC 无坐标，需从 DEM 读取
        try:
            epsg = get_epsg_from_file(amp_ref)
            log.info("EPSG 从 NISAR 幅度图读取: %d", epsg)
        except Exception:
            epsg = get_epsg_from_file(geogrid_dem)
            log.info("EPSG 从 DEM 读取: %d", epsg)

    _csmin = proc.get("chip_size_min_m")
    _csmax = proc.get("chip_size_max_m")

    geogrid_files = build_geogrid_inputs(
        dem_file=geogrid_dem, ssm_file=ssm_file, output_dir=geogrid_dir,
        chip_size_min_m=float(_csmin) if _csmin is not None else None,
        chip_size_max_m=float(_csmax) if _csmax is not None else None,
        search_range_m_yr=float(proc["search_range_m_yr"]),
        sps_files=sps_files,
    )

    # Step 6: Geogrid
    log.info("=" * 60)
    log.info("Step 6: runGeogrid（optical_flag=%d）", optical_flag)
    geogrid_run_info = run_geogrid_sar(
        meta_r=meta_r, meta_s=meta_s,
        epsg=epsg, geogrid_inputs=geogrid_files,
        working_dir=window_dir,
        optical_flag=optical_flag,   # ← RSLC=0, GSLC=1
    )

    # Step 7: autoRIFT
    log.info("=" * 60)
    log.info("Step 7: autoRIFT 特征追踪")
    vx0_for_error = geogrid_files["vx"] if has_prior_v else None
    vy0_for_error = geogrid_files["vy"] if has_prior_v else None

    results = run_autorift(
        ref_tif=amp_ref, sec_tif=amp_sec,
        window_dir=window_dir, output_dir=work_dir,
        pair_name=pair_name,
        #filter_type=proc.get("filter_type"),        # ← 移除 or "WAL" 默认回退
        #filter_width=int(proc.get("filter_width") or 21), # ← 保持安全回退（仅 S1 生效）
        oversample_ratio=int(proc.get("oversample_ratio", 64)),
        n_threads=int(proc.get("n_threads", 4)),
        save_offsets=bool(proc.get("save_offsets", True)),
        velocity_max_m_yr=float(proc.get("velocity_max_m_yr", 20000.0)),
        vx0_tif=vx0_for_error, vy0_tif=vy0_for_error,
        save_netcdf=bool(proc.get("output_netcdf", True)),
        date_ref=date_ref, date_sec=date_sec, epsg=epsg,
        geogrid_run_info=geogrid_run_info,  # ← 核心修复：注入 Geogrid 运行参数
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    log.info("加载配置: %s", args.config)
    cfg    = load_config(args.config)
    sensor = _detect_sensor_from_config(cfg)
    proc   = cfg["processing"]

    # ── 工作目录路由（与 v3 一致）────────────────────────────────────────────
    inp = cfg["input"]
    if sensor == 'S1':
        ref_path = inp["reference_zip"]
        sec_path = inp["secondary_zip"]
    else:
        ref_path = inp["reference_h5"]
        sec_path = inp["secondary_h5"]

    date_ref  = _date_from_file(ref_path, sensor)
    date_sec  = _date_from_file(sec_path, sensor)
    pair_name = proc.get("pair_name") or f"{date_ref}_{date_sec}"

    base_results_dir = proc.get("base_results_dir", "results")
    ensure_dir(base_results_dir)

    work_dir = str(Path(base_results_dir) / pair_name)
    ensure_dir(work_dir)

    log.info("=" * 60)
    log.info("传感器: %s  |  像对: %s", sensor, pair_name)
    log.info("工作目录: %s", work_dir)
    log.info("=" * 60)

    # ── 分发到传感器流程 ──────────────────────────────────────────────────────
    if sensor == 'S1':
        results = _run_s1(cfg, work_dir, args)
    elif sensor == 'NISAR':
        if args.list_bursts:
            log.warning("--list-bursts 仅适用于 S1，NISAR 不支持此选项。")
            _sys.exit(0)
        results = _run_nisar(cfg, work_dir, args)
    else:
        raise ValueError(f"未知传感器类型: {sensor}")

    # ── 完成摘要 ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✓ 全部完成！传感器=%s  像对=%s", sensor, pair_name)
    log.info("  速度场 GeoTIFF:  %s", results["velocity_tif"])
    if "offset_tif" in results:
        log.info("  偏移量:          %s", results["offset_tif"])
    if "netcdf" in results:
        log.info("  NetCDF CF-1.8:   %s", results["netcdf"])
    log.info(
        "  误差: Vx_err=%.2f  Vy_err=%.2f  V_err=%.2f  stable_count=%d  [m/yr]",
        results.get("vx_error",    float("nan")),
        results.get("vy_error",    float("nan")),
        results.get("v_error",     float("nan")),
        results.get("stable_count", 0),
    )

    # ── 中间文件清理 ──────────────────────────────────────────────────────────
    if not args.keep_intermediates:
        log.info("清理中间文件（保留 velocity_*.tif / velocity_*.nc）…")
        _cleanup_intermediates(work_dir)
    else:
        log.info("--keep-intermediates 已启用，跳过中间文件清理。")


if __name__ == "__main__":
    main()
