"""
geogrid_builder_optical.py — 构建光学 Geogrid 输入栅格并运行 GeogridOptical

与 S1 版本的核心差异：
  - 使用 GeogridOptical 而非 GeogridRadar
  - optical_flag=True：runGeogrid 走光学分支
  - meta_r/meta_s 使用 startingX/Y/XSize/YSize（地图坐标），而非雷达参数
  - repeatTime 由日期差计算（秒），而非 sensingStart 差值
  - chipSizeX0 默认 240m（官方光学默认值）

两种工作模式（与 S1 版本相同）：
  模式 A — SPS 参数文件（优先）
  模式 B — 均匀回退（默认）
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from osgeo import gdal

from .utils import (
    ensure_dir,
    get_raster_info,
    read_geotiff,
    write_geotiff,
)

gdal.UseExceptions()
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPS 重采样助手（与 S1 版本共享逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def _resample_to_dem(
    src_file: str,
    dem_file: str,
    resample_alg: str = "bilinear",
    out_dtype=np.float32,
    nodata_src: float = -32767.0,
) -> np.ndarray:
    info = get_raster_info(dem_file)
    warp_opts = gdal.WarpOptions(
        format="MEM",
        width=info["cols"],
        height=info["rows"],
        outputBounds=(info["x_min"], info["y_min"], info["x_max"], info["y_max"]),
        resampleAlg=resample_alg,
        dstSRS=info["projection"],
        srcNodata=nodata_src,
        dstNodata=nodata_src,
    )
    ds  = gdal.Warp("", src_file, options=warp_opts)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(out_dtype)
    ds  = None
    if np.issubdtype(out_dtype, np.floating):
        arr[arr == nodata_src] = np.nan
    return arr


def _sps_path(sps_files: Optional[Dict], key: str) -> Optional[str]:
    if not sps_files:
        return None
    p = sps_files.get(key)
    if not p:
        return None
    path = Path(p)
    if not path.exists():
        log.warning("SPS 文件未找到（已跳过）: %s = %s", key, p)
        return None
    return str(path.resolve())


def _resample_ssm(ssm_file: str, dem_file: str) -> np.ndarray:
    info = get_raster_info(dem_file)
    warp_opts = gdal.WarpOptions(
        format="MEM",
        width=info["cols"],
        height=info["rows"],
        outputBounds=(info["x_min"], info["y_min"], info["x_max"], info["y_max"]),
        resampleAlg="near",
        dstSRS=info["projection"],
    )
    ds  = gdal.Warp("", ssm_file, options=warp_opts)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    arr[arr > 0] = 1
    ds  = None
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2. 构建 Geogrid 输入栅格
# ─────────────────────────────────────────────────────────────────────────────

def build_geogrid_inputs_optical(
    dem_file: str,
    ssm_file: Optional[str],
    output_dir: str,
    chip_size_min_m: Optional[float] = 240.0,
    chip_size_max_m: Optional[float] = 960.0,
    search_range_m_yr: float = 3000.0,
    sps_files: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """从 DEM + SSM（+ 可选 SPS 参数文件）构建光学 Geogrid 输入栅格。

    光学模式与 SAR 模式的 Geogrid 输入格式完全相同；
    差异在于 Geogrid 内部使用 GeogridOptical 分支（由 optical_flag=1 控制）。

    Returns
    -------
    dict  键名与 runGeogrid 参数名一一对应：
          dem, dhdx, dhdy, ssm, vx, vy, srx, sry, csminx, csminy, csmaxx, csmaxy
    """
    ensure_dir(output_dir)
    out = Path(output_dir).resolve()

    log.info("读取 DEM: %s", dem_file)
    dem_arr, gt, proj = read_geotiff(dem_file)
    dem_arr = dem_arr.astype(np.float32)
    rows, cols = dem_arr.shape
    dx = abs(gt[1])
    dy = abs(gt[5])
    log.info("DEM 尺寸: %d × %d  分辨率: %.1f m × %.1f m", rows, cols, dx, dy)

    # ── dhdx / dhdy ──────────────────────────────────────────────────────────
    dhdx_src = _sps_path(sps_files, "dhdx_file")
    dhdy_src = _sps_path(sps_files, "dhdy_file")

    if dhdx_src and dhdy_src:
        log.info("dhdx/dhdy: 使用 SPS 文件")
        dhdx = np.nan_to_num(_resample_to_dem(dhdx_src, dem_file), nan=0.0).astype(np.float32)
        dhdy = np.nan_to_num(_resample_to_dem(dhdy_src, dem_file), nan=0.0).astype(np.float32)
    else:
        log.info("dhdx/dhdy: 从 DEM 计算梯度")
        dhdx = (np.gradient(dem_arr, axis=1) / dx).astype(np.float32)
        dhdy = (np.gradient(dem_arr, axis=0) / dy).astype(np.float32)
        dhdx[np.abs(dhdx) > 5.0] = 0.0
        dhdy[np.abs(dhdy) > 5.0] = 0.0

    # ── chip size ─────────────────────────────────────────────────────────────
    _CS_NODATA = 0.0
    csminx_src = _sps_path(sps_files, "chip_size_min_x_file")
    csminy_src = _sps_path(sps_files, "chip_size_min_y_file")
    csmaxx_src = _sps_path(sps_files, "chip_size_max_x_file")
    csmaxy_src = _sps_path(sps_files, "chip_size_max_y_file")

    if chip_size_min_m is not None:
        _csmin = float(chip_size_min_m)
        _csmax = float(chip_size_max_m) if chip_size_max_m is not None else _csmin * 4.0
        log.info("chip_size: 均匀值 (min=%.0f m, max=%.0f m)", _csmin, _csmax)
        csminx = np.full((rows, cols), _csmin, dtype=np.float32)
        csminy = np.full((rows, cols), _csmin, dtype=np.float32)
        csmaxx = np.full((rows, cols), _csmax, dtype=np.float32)
        csmaxy = np.full((rows, cols), _csmax, dtype=np.float32)

    elif csminx_src and csminy_src:
        log.info("chip_size: 使用 SPS 文件 (nodata=0)")
        csminx = np.where(
            np.isfinite(r := _resample_to_dem(csminx_src, dem_file, nodata_src=_CS_NODATA)) & (r > 0),
            r, 240.0
        ).astype(np.float32)
        csminy = np.where(
            np.isfinite(r := _resample_to_dem(csminy_src, dem_file, nodata_src=_CS_NODATA)) & (r > 0),
            r, 240.0
        ).astype(np.float32)
        csmaxx = (np.where(
            np.isfinite(r := _resample_to_dem(csmaxx_src, dem_file, nodata_src=_CS_NODATA)) & (r > 0),
            r, 960.0
        ).astype(np.float32) if csmaxx_src else np.clip(csminx * 4.0, csminx, None))
        csmaxy = (np.where(
            np.isfinite(r := _resample_to_dem(csmaxy_src, dem_file, nodata_src=_CS_NODATA)) & (r > 0),
            r, 960.0
        ).astype(np.float32) if csmaxy_src else np.clip(csminy * 4.0, csminy, None))
    else:
        raise ValueError(
            "chip size 未配置：请在 config.yaml 中设置 chip_size_min_m/chip_size_max_m "
            "或提供 SPS chip size 文件。"
        )

    # ── 搜索范围 ──────────────────────────────────────────────────────────────
    srx_src = _sps_path(sps_files, "search_range_x_file")
    sry_src = _sps_path(sps_files, "search_range_y_file")

    if srx_src and sry_src:
        log.info("search_range: 使用 SPS 文件")
        srx_raw = _resample_to_dem(srx_src, dem_file, nodata_src=_CS_NODATA)
        sry_raw = _resample_to_dem(sry_src, dem_file, nodata_src=_CS_NODATA)
        srx = np.where(np.isfinite(srx_raw) & (srx_raw > 0), srx_raw, search_range_m_yr).astype(np.float32)
        sry = np.where(np.isfinite(sry_raw) & (sry_raw > 0), sry_raw, search_range_m_yr).astype(np.float32)
        srx = np.clip(srx, 0, 32000)
        sry = np.clip(sry, 0, 32000)
    else:
        log.info("search_range: %.0f m/yr（均匀值）", search_range_m_yr)
        srx = np.full((rows, cols), search_range_m_yr, dtype=np.float32)
        sry = np.full((rows, cols), search_range_m_yr, dtype=np.float32)

    # ── 参考速度场 vx0/vy0 ────────────────────────────────────────────────────
    vx0_src = _sps_path(sps_files, "vx0_file")
    vy0_src = _sps_path(sps_files, "vy0_file")

    if vx0_src and vy0_src:
        log.info("vx0/vy0: 使用 SPS 参考速度场")
        vx_arr = np.nan_to_num(_resample_to_dem(vx0_src, dem_file, nodata_src=-32767.0), nan=0.0).astype(np.float32)
        vy_arr = np.nan_to_num(_resample_to_dem(vy0_src, dem_file, nodata_src=-32767.0), nan=0.0).astype(np.float32)
    else:
        log.info("vx0/vy0: 全零（均匀回退）")
        vx_arr = np.zeros((rows, cols), dtype=np.float32)
        vy_arr = np.zeros((rows, cols), dtype=np.float32)

    # ── 稳定地表掩膜 ──────────────────────────────────────────────────────────
    if ssm_file and Path(ssm_file).exists():
        log.info("读取 SSM: %s", ssm_file)
        ssm_arr = _resample_ssm(ssm_file, dem_file)
    else:
        log.info("未提供 SSM，使用全零掩膜")
        ssm_arr = np.zeros((rows, cols), dtype=np.uint8)

    # ── 写出所有栅格 ──────────────────────────────────────────────────────────
    files = {
        "dem":    str(Path(dem_file).resolve()),
        "dhdx":   str((out / "dhdx.tif").resolve()),
        "dhdy":   str((out / "dhdy.tif").resolve()),
        "csminx": str((out / "csminx.tif").resolve()),
        "csminy": str((out / "csminy.tif").resolve()),
        "csmaxx": str((out / "csmaxx.tif").resolve()),
        "csmaxy": str((out / "csmaxy.tif").resolve()),
        "srx":    str((out / "srx.tif").resolve()),
        "sry":    str((out / "sry.tif").resolve()),
        "vx":     str((out / "vx.tif").resolve()),
        "vy":     str((out / "vy.tif").resolve()),
        "ssm":    str((out / "ssm.tif").resolve()),
    }

    write_geotiff(files["dhdx"],   [dhdx],    gt, proj, nodata=0.0,      band_names=["dhdx"])
    write_geotiff(files["dhdy"],   [dhdy],    gt, proj, nodata=0.0,      band_names=["dhdy"])
    write_geotiff(files["csminx"], [csminx],  gt, proj,                  band_names=["csminx"])
    write_geotiff(files["csminy"], [csminy],  gt, proj,                  band_names=["csminy"])
    write_geotiff(files["csmaxx"], [csmaxx],  gt, proj,                  band_names=["csmaxx"])
    write_geotiff(files["csmaxy"], [csmaxy],  gt, proj,                  band_names=["csmaxy"])
    write_geotiff(files["srx"],    [srx],     gt, proj,                  band_names=["srx"])
    write_geotiff(files["sry"],    [sry],     gt, proj,                  band_names=["sry"])
    write_geotiff(files["vx"],     [vx_arr],  gt, proj, nodata=-32767.0, band_names=["vx"])
    write_geotiff(files["vy"],     [vy_arr],  gt, proj, nodata=-32767.0, band_names=["vy"])
    write_geotiff(files["ssm"],    [ssm_arr], gt, proj, nodata=255,      band_names=["ssm"])

    log.info("光学 Geogrid 输入栅格已写入: %s", output_dir)
    return files


# ─────────────────────────────────────────────────────────────────────────────
# 3. 运行 GeogridOptical
# ─────────────────────────────────────────────────────────────────────────────

def run_geogrid_optical(
    meta_r,
    meta_s,
    epsg: int,
    geogrid_inputs: Dict[str, str],
    working_dir: str = ".",
) -> dict:
    """调用 GeogridOptical 生成 window_*.tif。

    与 S1 版本的关键差异：
      - optical_flag=1（使用 GeogridOptical 分支）
      - meta_r/meta_s 携带地图坐标参数（startingX/Y/XSize/YSize）
      - repeatTime 由 meta 中的 time 字段（YYYYMMDD）计算

    Parameters
    ----------
    meta_r, meta_s : optical_processor.load_optical_metadata_pair() 的返回值
    """
    try:
        from geogrid import GeogridOptical
    except ImportError:
        raise ImportError("geogrid 未安装，请确保 autoRIFT 环境已激活。")

    orig_dir = os.getcwd()
    ensure_dir(working_dir)

    # 绝对路径化
    geogrid_inputs_abs = {
        k: str(Path(v).resolve()) if v and Path(v).exists() else v
        for k, v in geogrid_inputs.items()
    }

    os.chdir(working_dir)
    try:
        log.info("运行 GeogridOptical (EPSG:%d)...", epsg)

        # 计算时间差（秒）
        d0 = date(int(meta_r.time[:4]), int(meta_r.time[4:6]), int(meta_r.time[6:8]))
        d1 = date(int(meta_s.time[:4]), int(meta_s.time[4:6]), int(meta_s.time[6:8]))
        repeat_time = (d1 - d0).total_seconds()
        log.info("时间跨度: %d 天 (%.0f 秒)", (d1 - d0).days, repeat_time)

        dem_info = gdal.Info(geogrid_inputs_abs["dem"], format="json")

        obj = GeogridOptical()

        # 地图坐标参数（光学 meta 特有）
        obj.startingX       = meta_r.startingX
        obj.startingY       = meta_r.startingY
        obj.XSize           = meta_r.XSize
        obj.YSize           = meta_r.YSize
        obj.numberOfLines   = meta_r.numberOfLines
        obj.numberOfSamples = meta_r.numberOfSamples
        obj.repeatTime      = repeat_time
        obj.nodata_out      = -32767
        obj.chipSizeX0      = 240           # 官方光学默认值
        obj.gridSpacingX    = dem_info["geoTransform"][1]

         # ── 新增：确保输出尺寸为 32 的整数倍 ──────────────────────
        # 计算对齐后的尺寸（向下取整）
        original_rows = meta_r.numberOfLines
        original_cols = meta_r.numberOfSamples
        
        aligned_rows = (original_rows // 32) * 32
        aligned_cols = (original_cols // 32) * 32
        
        if aligned_rows != original_rows or aligned_cols != original_cols:
            log.info(
                "调整 Geogrid 尺寸: %dx%d → %dx%d (对齐到 32 的倍数)",
                original_rows, original_cols, aligned_rows, aligned_cols
            )
        obj.numberOfLines = aligned_rows
        obj.numberOfSamples = aligned_cols
    # ───────────────────────────────────────────────────────

        # 参考影像路径（Geogrid 读取其 GeoTransform）
        obj.dat1name = meta_r.filename

        # 地形文件
        obj.demname   = geogrid_inputs_abs["dem"]
        obj.dhdxname  = geogrid_inputs_abs.get("dhdx", "")
        obj.dhdyname  = geogrid_inputs_abs.get("dhdy", "")

        # 参考速度场
        obj.vxname  = geogrid_inputs_abs.get("vx", "")
        obj.vyname  = geogrid_inputs_abs.get("vy", "")

        # 搜索范围
        obj.srxname  = geogrid_inputs_abs.get("srx", "")
        obj.sryname  = geogrid_inputs_abs.get("sry", "")

        # Chip size
        obj.csminxname = geogrid_inputs_abs.get("csminx", "")
        obj.csminyname = geogrid_inputs_abs.get("csminy", "")
        obj.csmaxxname = geogrid_inputs_abs.get("csmaxx", "")
        obj.csmaxyname = geogrid_inputs_abs.get("csmaxy", "")

        # 稳定地表掩膜
        obj.ssmname = geogrid_inputs_abs.get("ssm", "")

        # 输出文件（与 S1 版本完全相同命名）
        obj.winlocname   = "window_location.tif"
        obj.winoffname   = "window_offset.tif"
        obj.winsrname    = "window_search_range.tif"
        obj.wincsminname = "window_chip_size_min.tif"
        obj.wincsmaxname = "window_chip_size_max.tif"
        obj.winssmname   = "window_stable_surface_mask.tif"
        obj.winro2vxname = "window_rdr_off2vel_x_vec.tif"
        obj.winro2vyname = "window_rdr_off2vel_y_vec.tif"
        obj.winsfname    = "window_scale_factor.tif"

        obj.runGeogrid()
        gdal.AllRegister()

        run_info = {
            "chipsizex0":  obj.chipSizeX0,
            "gridspacingx": obj.gridSpacingX,
            "vxname":      geogrid_inputs_abs.get("vx", ""),
            "vyname":      geogrid_inputs_abs.get("vy", ""),
            "xoff":        obj.pOff,
            "yoff":        obj.lOff,
            "xcount":      obj.pCount,
            "ycount":      obj.lCount,
            "dt":          obj.repeatTime,
            "epsg":        epsg,
            "XPixelSize":  obj.X_res,
            "YPixelSize":  obj.Y_res,
            "cen_lat":     obj.cen_lat,
            "cen_lon":     obj.cen_lon,
        }
        log.info("GeogridOptical 完成。输出格网: %s × %s", run_info["ycount"], run_info["xcount"])

    finally:
        os.chdir(orig_dir)

    _ensure_scale_factor(working_dir, run_info)
    return run_info


def _ensure_scale_factor(working_dir: str, run_info: dict) -> None:
    sf_path  = Path(working_dir) / "window_scale_factor.tif"
    loc_path = Path(working_dir) / "window_location.tif"
    if sf_path.exists() or not loc_path.exists():
        return
    arr, gt, proj = read_geotiff(str(loc_path))
    ycount = run_info.get("ycount", arr.shape[-2])
    xcount = run_info.get("xcount", arr.shape[-1])
    ones   = np.ones((ycount, xcount), dtype=np.float64)
    write_geotiff(str(sf_path), [ones, ones], gt, proj,
                  band_names=["scale_x", "scale_y"])
    log.info("已生成 window_scale_factor.tif（全1占位）")
