"""
geogrid_builder.py — 构建 Geogrid 输入栅格并运行 Geogrid

两种工作模式（自动检测，向后兼容）：

  模式 A — SPS 参数文件（优先）
    提供 sps_files 中的任意参数栅格（dhdx/dhdy/sr/csmin/csmax/vx0/vy0），
    对应字段将被重采样（双线性/最近邻）到 DEM 格网，替代均匀占位值。
    覆盖整个南极的 SPS_0120m_*.tif 文件均支持此路径。

  模式 B — 均匀回退（默认）
    未提供 SPS 文件时，按 chip_size_min/max_m 和 search_range_m_yr 生成
    全场均匀栅格，行为与 v3 旧版完全一致。
"""

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, osr

from .utils import (
    read_geotiff,
    write_geotiff,
    get_raster_info,
    ensure_dir,
)

gdal.UseExceptions()
log = logging.getLogger(__name__)

# testGeogrid.py in the vendored HyP3 implementation sets chipSizeX0 to 240 m.
# Keep the DEM-only fallback and the user-facing validation tied to that same
# value, rather than accepting arbitrary values that autoRIFT cannot use as a
# nested ChipSize0X hierarchy.
DEFAULT_BASE_CHIP_SIZE_M = 240.0
SUPPORTED_CHIP_SCALES = (1, 2, 4, 8)


def prepare_geogrid_dem(
    dem_file: str,
    output_dir: str,
    output_resolution_m: Optional[float] = None,
    reuse_existing: bool = True,
) -> str:
    """Return the DEM used by Geogrid, optionally resampled to a target spacing.

    The S1/NISAR preprocessors keep using ``dem_file`` at its native resolution.
    Only the Geogrid branch uses the returned DEM.  Thus a user can request a
    30/60/120 m velocity product without degrading the DEM used for RSLC
    co-registration.

    A requested resolution requires a projected DEM whose linear units are
    metres.  Degree-based geographic DEMs cannot express ``output_resolution_m``.
    """
    source = str(Path(dem_file).resolve())
    if output_resolution_m is None:
        return source

    target_res = float(output_resolution_m)
    if not np.isfinite(target_res) or target_res <= 0:
        raise ValueError("processing.output_resolution_m 必须是正数（单位：m），或设为 null")

    info = get_raster_info(source)
    gt = info["geo_transform"]
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise ValueError("暂不支持带旋转项的 DEM；请先 north-up 重投影后再运行")

    srs = osr.SpatialReference(info["projection"])
    if not srs.IsProjected() or not (0.9 <= srs.GetLinearUnits() <= 1.1):
        raise ValueError(
            "要使用 processing.output_resolution_m，DEM 必须是以米为单位的投影坐标系 "
            "（例如南极 EPSG:3031），不能使用经纬度 DEM。"
        )

    src_xres = abs(float(info["x_res"]))
    src_yres = abs(float(info["y_res"]))
    if math.isclose(src_xres, target_res, rel_tol=0.0, abs_tol=1e-6) and math.isclose(
        src_yres, target_res, rel_tol=0.0, abs_tol=1e-6
    ):
        log.info("Geogrid 输出分辨率已是 %.3f m，直接使用输入 DEM", target_res)
        return source

    ensure_dir(output_dir)
    resolution_tag = f"{target_res:g}".replace(".", "p")
    target = Path(output_dir) / f"dem_geogrid_{resolution_tag}m.tif"
    if reuse_existing and target.exists():
        cached = get_raster_info(str(target))
        if math.isclose(abs(cached["x_res"]), target_res, rel_tol=0.0, abs_tol=1e-6) and math.isclose(
            abs(cached["y_res"]), target_res, rel_tol=0.0, abs_tol=1e-6
        ):
            log.info("复用已存在的 Geogrid DEM: %s", target)
            return str(target.resolve())

    # Keep the upper-left corner fixed and expand only the lower/right edge when
    # needed.  This produces exact requested metre spacing without shifting all
    # output nodes relative to the source DEM.
    width = int(math.ceil((info["x_max"] - info["x_min"]) / target_res))
    height = int(math.ceil((info["y_max"] - info["y_min"]) / target_res))
    bounds = (
        info["x_min"],
        info["y_max"] - height * target_res,
        info["x_min"] + width * target_res,
        info["y_max"],
    )
    src_ds = gdal.Open(source, gdal.GA_ReadOnly)
    src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
    src_ds = None
    warp_kwargs = dict(
        format="GTiff",
        outputBounds=bounds,
        xRes=target_res,
        yRes=target_res,
        dstSRS=info["projection"],
        resampleAlg="bilinear",
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    if src_nodata is not None:
        warp_kwargs["srcNodata"] = src_nodata
        warp_kwargs["dstNodata"] = src_nodata
    options = gdal.WarpOptions(**warp_kwargs)
    log.info("重采样 Geogrid DEM: %.3f m → %.3f m", src_xres, target_res)
    warped = gdal.Warp(str(target), source, options=options)
    if warped is None:
        raise RuntimeError(f"无法生成目标分辨率 Geogrid DEM: {target}")
    warped = None
    return str(target.resolve())


def _normalise_uniform_chip_sizes(
    chip_size_min_m: Optional[float],
    chip_size_max_m: Optional[float],
    base_chip_size_m: float = DEFAULT_BASE_CHIP_SIZE_M,
) -> Tuple[float, float]:
    """Map DEM-only chip settings to supported 1/2/4/8 base-chip levels.

    The values are upper-rounded deliberately: a user asking for at least
    600 m gets the 960 m level instead of an unexpectedly smaller 480 m window.
    """
    raw_min = base_chip_size_m if chip_size_min_m is None else float(chip_size_min_m)
    raw_max = raw_min * 4.0 if chip_size_max_m is None else float(chip_size_max_m)
    if not np.isfinite(raw_min) or not np.isfinite(raw_max) or raw_min <= 0 or raw_max <= 0:
        raise ValueError("chip_size_min_m / chip_size_max_m 必须为正数，或设为 null")
    raw_max = max(raw_max, raw_min)
    allowed = [base_chip_size_m * scale for scale in SUPPORTED_CHIP_SCALES]

    def snap_up(value: float) -> float:
        for level in allowed:
            if value <= level:
                return level
        raise ValueError(
            f"请求 ChipSize={value:.0f} m 超过当前管线支持的最大 {allowed[-1]:.0f} m；"
            "请裁小 AOI，或扩展 autorift_runner 的嵌套层级。"
        )

    minimum = snap_up(raw_min)
    maximum = max(minimum, snap_up(raw_max))
    if (minimum, maximum) != (raw_min, raw_max):
        log.warning(
            "ChipSize 配置 %.0f–%.0f m 已对齐为 %.0f–%.0f m；"
            "当前 Geogrid 基础 ChipSizeX0=%.0f m，只支持 1/2/4/8 倍层级。",
            raw_min, raw_max, minimum, maximum, base_chip_size_m,
        )
    return minimum, maximum


# ── 1. 通用 SPS 重采样助手 ────────────────────────────────────────────────────

def _resample_to_dem(
    src_file: str,
    dem_file: str,
    resample_alg: str = "bilinear",
    out_dtype=np.float32,
    nodata_src: float = -32767.0,
) -> np.ndarray:
    """将任意栅格重采样（默认双线性）到 DEM 格网，返回 float32 数组。

    Parameters
    ----------
    src_file     : 源栅格路径（可为全南极覆盖的 SPS 文件）
    dem_file     : 目标格网参考（复用其 extent / resolution / projection）
    resample_alg : 'bilinear'（连续字段）或 'near'（分类/掩膜）
    out_dtype    : 返回数组类型，默认 float32
    nodata_src   : 源栅格的 NoData 值，重采样后转为 np.nan / 0
    """
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
    ds = gdal.Warp("", src_file, options=warp_opts)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(out_dtype)
    ds = None
    # 将 nodata 转为 nan（浮点）或 0（整型）
    if np.issubdtype(out_dtype, np.floating):
        arr[arr == nodata_src] = np.nan
    return arr


def _sps_path(sps_files: Optional[Dict], key: str) -> Optional[str]:
    """安全取出 SPS 文件路径；不存在 / null / 文件缺失时返回 None。"""
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


# ── 2. 构建 Geogrid 输入栅格 ───────────────────────────────────────────────────

def build_geogrid_inputs(
    dem_file: str,
    ssm_file: Optional[str],
    output_dir: str,
    chip_size_min_m: Optional[float] = DEFAULT_BASE_CHIP_SIZE_M,
    chip_size_max_m: Optional[float] = DEFAULT_BASE_CHIP_SIZE_M * 4.0,
    search_range_m_yr: float = 3000.0,
    sps_files: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """从 DEM + SSM（+ 可选 SPS 参数文件）构建 Geogrid 所需全部输入栅格。

    Parameters
    ----------
    sps_files : dict, optional
        键名对应 config.yaml sps_params 节，值为文件路径或 None。
        提供某字段时替代对应的均匀回退；未提供字段保持均匀值。
        例：{'dhdx_file': '/path/SPS_0120m_dhdx.tif', 'dhdy_file': ...}

    Returns
    -------
    dict  键名与 runGeogrid 参数名一一对应
    """
    ensure_dir(output_dir)
    out = Path(output_dir).resolve()

    # ── 读取 DEM ─────────────────────────────────────────────────────────────
    log.info("读取 DEM: %s", dem_file)
    dem_arr, gt, proj = read_geotiff(dem_file)
    dem_arr = dem_arr.astype(np.float32)
    rows, cols = dem_arr.shape
    dx = abs(gt[1])
    dy = abs(gt[5])
    log.info("DEM 尺寸: %d × %d  分辨率: %.1f m × %.1f m", rows, cols, dx, dy)

    # ─────────────────────────────────────────────────────────────────────────
    # dhdx / dhdy  — 优先用 SPS，否则从 DEM 计算梯度
    # ─────────────────────────────────────────────────────────────────────────
    dhdx_src = _sps_path(sps_files, "dhdx_file")
    dhdy_src = _sps_path(sps_files, "dhdy_file")

    if dhdx_src and dhdy_src:
        log.info("dhdx/dhdy: 使用 SPS 文件")
        dhdx = _resample_to_dem(dhdx_src, dem_file)
        dhdy = _resample_to_dem(dhdy_src, dem_file)
        # 将 nan 归零（边界外无效区）
        dhdx = np.nan_to_num(dhdx, nan=0.0).astype(np.float32)
        dhdy = np.nan_to_num(dhdy, nan=0.0).astype(np.float32)
    else:
        log.info("dhdx/dhdy: 从 DEM 计算梯度（均匀回退）")
        dhdx = (np.gradient(dem_arr, axis=1) / dx).astype(np.float32)
        dhdy = (np.gradient(dem_arr, axis=0) / dy).astype(np.float32)
        dhdx[np.abs(dhdx) > 5.0] = 0.0
        dhdy[np.abs(dhdy) > 5.0] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # chip size min / max
    #
    # 优先级（对齐 hyp3 chip_size override 逻辑）：
    #   1. config 中 chip_size_min_m / chip_size_max_m 为非 null
    #      → 使用均匀值，完全跳过 SPS chip size tif（推荐，避免 ChipSize0 倍数校验失败）
    #   2. chip_size_min_m = null + SPS tif 存在
    #      → 从 SPS tif 重采样（nodata=0，双线性）
    #   3. 两者均无
    #      → 使用安全的 240–960 m DEM-only 回退；这使“只有 DEM”的配置可运行。
    #
    # 【为什么 SPS tif 不能直接用？】
    #   autoRIFT 要求所有 ChipSizeMinX 值均为 ChipSize0（最小有效值）的整数倍。
    #   SPS 米制值经过雷达像素转换后（如 240m→64px, 480m→128px, 但也可能出现 96px、
    #   192px 等非 64 倍数的值），无法保证该关系 → 抛出 "chip sizes must be even
    #   integers of ChipSize0" 错误。
    #   均匀值（如 240m:960m = 1:4）经 Geogrid 转换后保持精确整数倍关系，安全。
    #
    # 【SPS chip tif NoData】UInt16 chip size 文件的 NoData = 0（不是 -32767）。
    # ─────────────────────────────────────────────────────────────────────────
    _CS_NODATA = 0.0

    csminx_src = _sps_path(sps_files, "chip_size_min_x_file")
    csminy_src = _sps_path(sps_files, "chip_size_min_y_file")
    csmaxx_src = _sps_path(sps_files, "chip_size_max_x_file")
    csmaxy_src = _sps_path(sps_files, "chip_size_max_y_file")

    # ── 均匀值优先（config 中 chip_size_min/max_m 非 null）──────────────────
    if chip_size_min_m is not None:
        _csmin, _csmax = _normalise_uniform_chip_sizes(
            chip_size_min_m,
            chip_size_max_m,
        )
        log.info(
            "chip_size: 使用 config 均匀值（min=%.0f m, max=%.0f m）—— SPS chip size tif 已跳过",
            _csmin, _csmax,
        )
        csminx = np.full((rows, cols), _csmin, dtype=np.float32)
        csminy = np.full((rows, cols), _csmin, dtype=np.float32)
        csmaxx = np.full((rows, cols), _csmax, dtype=np.float32)
        csmaxy = np.full((rows, cols), _csmax, dtype=np.float32)

    # ── 使用 SPS chip size tif（chip_size_min_m = null）─────────────────────
    elif csminx_src and csminy_src:
        log.info("chip_size: chip_size_min_m=null → 使用 SPS 文件（UInt16，nodata=0）")
        log.warning(
            "注意：SPS chip size tif 转换后的像素值可能不满足 ChipSize0 倍数约束，"
            "若出现 'chip sizes must be even integers' 错误，请在 config 中显式设置 "
            "chip_size_min_m 以强制使用均匀值。"
        )
        csminx_raw = _resample_to_dem(csminx_src, dem_file, nodata_src=_CS_NODATA)
        csminy_raw = _resample_to_dem(csminy_src, dem_file, nodata_src=_CS_NODATA)
        csminx = np.where(np.isfinite(csminx_raw) & (csminx_raw > 0),
                          csminx_raw, 240.0).astype(np.float32)
        csminy = np.where(np.isfinite(csminy_raw) & (csminy_raw > 0),
                          csminy_raw, 240.0).astype(np.float32)
        csminx = np.clip(csminx, dx, None)
        csminy = np.clip(csminy, dy, None)
        log.info("  chip_size_min SPS: x=[%.0f, %.0f] m  y=[%.0f, %.0f] m",
                 float(csminx.min()), float(csminx.max()),
                 float(csminy.min()), float(csminy.max()))

        if csmaxx_src and csmaxy_src:
            csmaxx_raw = _resample_to_dem(csmaxx_src, dem_file, nodata_src=_CS_NODATA)
            csmaxy_raw = _resample_to_dem(csmaxy_src, dem_file, nodata_src=_CS_NODATA)
            csmaxx = np.where(np.isfinite(csmaxx_raw) & (csmaxx_raw > 0),
                              csmaxx_raw, 960.0).astype(np.float32)
            csmaxy = np.where(np.isfinite(csmaxy_raw) & (csmaxy_raw > 0),
                              csmaxy_raw, 960.0).astype(np.float32)
            csmaxx = np.clip(csmaxx, csminx, None)
            csmaxy = np.clip(csmaxy, csminy, None)
            log.info("  chip_size_max SPS: x=[%.0f, %.0f] m  y=[%.0f, %.0f] m",
                     float(csmaxx.min()), float(csmaxx.max()),
                     float(csmaxy.min()), float(csmaxy.max()))
        else:
            csmaxx = np.clip(csminx * 4.0, csminx, None)
            csmaxy = np.clip(csminy * 4.0, csminy, None)

    else:
        _csmin, _csmax = _normalise_uniform_chip_sizes(None, None)
        log.info(
            "chip_size: 未提供 SPS chip tif 或固定值，使用 DEM-only 自动回退 "
            "（min=%.0f m, max=%.0f m）",
            _csmin, _csmax,
        )
        csminx = np.full((rows, cols), _csmin, dtype=np.float32)
        csminy = np.full((rows, cols), _csmin, dtype=np.float32)
        csmaxx = np.full((rows, cols), _csmax, dtype=np.float32)
        csmaxy = np.full((rows, cols), _csmax, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 搜索范围  — SPS tif 优先；config search_range_m_yr 为回退值。
    # aws3_tif 的 vx/vySearchRange 是 Int16，NoData=32767（不是 0）。
    # ─────────────────────────────────────────────────────────────────────────
    srx_src = _sps_path(sps_files, "search_range_x_file")
    sry_src = _sps_path(sps_files, "search_range_y_file")

    if srx_src and sry_src:
        _SEARCH_NODATA = 32767.0
        log.info("search_range: 使用 SPS 文件（Int16，nodata=32767，config 值作为回退）")
        srx_raw = _resample_to_dem(srx_src, dem_file, nodata_src=_SEARCH_NODATA)
        sry_raw = _resample_to_dem(sry_src, dem_file, nodata_src=_SEARCH_NODATA)
        srx = np.where(np.isfinite(srx_raw) & (srx_raw > 0),
                       srx_raw, search_range_m_yr).astype(np.float32)
        sry = np.where(np.isfinite(sry_raw) & (sry_raw > 0),
                       sry_raw, search_range_m_yr).astype(np.float32)
        srx = np.clip(srx, 0, 32000)
        sry = np.clip(sry, 0, 32000)
        log.info("  search_range: x=[%.0f, %.0f] m/yr  y=[%.0f, %.0f] m/yr",
                 float(srx.min()), float(srx.max()),
                 float(sry.min()), float(sry.max()))
    else:
        log.info("search_range: %.0f m/yr（config 均匀值）", search_range_m_yr)
        srx = np.full((rows, cols), search_range_m_yr, dtype=np.float32)
        sry = np.full((rows, cols), search_range_m_yr, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 参考速度场 vx0 / vy0  — 优先用 SPS，否则全零
    # ─────────────────────────────────────────────────────────────────────────
    vx0_src = _sps_path(sps_files, "vx0_file")
    vy0_src = _sps_path(sps_files, "vy0_file")

    if vx0_src and vy0_src:
        log.info("vx0/vy0: 使用 SPS 参考速度场（将改善搜索偏移量并启用误差估计）")
        vx_arr = np.nan_to_num(_resample_to_dem(vx0_src, dem_file, nodata_src=-32767.0), nan=0.0).astype(np.float32)
        vy_arr = np.nan_to_num(_resample_to_dem(vy0_src, dem_file, nodata_src=-32767.0), nan=0.0).astype(np.float32)
    else:
        log.info("vx0/vy0: 全零（均匀回退，误差将按稳定地表标准差估计）")
        vx_arr = np.zeros((rows, cols), dtype=np.float32)
        vy_arr = np.zeros((rows, cols), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 稳定地表掩膜 SSM
    # ─────────────────────────────────────────────────────────────────────────
    if ssm_file and Path(ssm_file).exists():
        log.info("读取 SSM: %s", ssm_file)
        ssm_arr = _resample_ssm(ssm_file, dem_file)
    else:
        log.info("未提供 SSM，使用全零掩膜")
        ssm_arr = np.zeros((rows, cols), dtype=np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # 写出所有栅格（全部使用绝对路径）
    # ─────────────────────────────────────────────────────────────────────────
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

    # ── 打印使用模式摘要 ──────────────────────────────────────────────────────
    sps_used = [k for k in (sps_files or {}) if _sps_path(sps_files, k)]
    if sps_used:
        log.info("SPS 模式：已使用 %d 个空间变化参数文件", len(sps_used))
    else:
        log.info("均匀回退模式：chip_size=[%.0f,%.0f]m  search_range=%.0f m/yr",
                 float(csminx[0, 0]), float(csmaxx[0, 0]), search_range_m_yr)

    log.info("Geogrid 输入栅格已写入: %s", output_dir)
    return files


# ── 3. SSM 重采样（保持原实现）───────────────────────────────────────────────

def _resample_ssm(ssm_file: str, dem_file: str) -> np.ndarray:
    """将 SSM 重采样（最近邻）到 DEM 格网并二值化。"""
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


# ── 4. 运行 Geogrid ───────────────────────────────────────────────────────────

def run_geogrid_sar(
    meta_r,
    meta_s,
    epsg: int,
    geogrid_inputs: Dict[str, str],
    working_dir: str = ".",
    optical_flag: int = 0,            # ← 新增：0=SAR 斜距, 1=地理编码(GSLC/光学)
) -> dict:
    """调用 hyp3_autorift.vend.testGeogrid.runGeogrid 生成 window_*.tif。

    Parameters
    ----------
    optical_flag : int
        0 → SAR 斜距几何（S1, NISAR RSLC）
        1 → 地理编码几何（NISAR GSLC, Sentinel-2, Landsat）
    """
    from hyp3_autorift.vend.testGeogrid import runGeogrid

    orig_dir = os.getcwd()
    ensure_dir(working_dir)

    geogrid_inputs_abs = {}
    for k, v in geogrid_inputs.items():
        p = Path(v)
        geogrid_inputs_abs[k] = str(p.resolve()) if (p.exists() and not p.is_absolute()) else v

    os.chdir(working_dir)

    try:
        log.info("运行 Geogrid（EPSG:%d, optical_flag=%d）...", epsg, optical_flag)
        geogrid_run_info = runGeogrid(
            info=meta_r,
            info1=meta_s,
            optical_flag=optical_flag,    # ← 原本硬编码 0，现在由调用方传入
            epsg=epsg,
            **geogrid_inputs_abs,
        )
        gdal.AllRegister()
        ycount = geogrid_run_info.get("ycount", "?")
        xcount = geogrid_run_info.get("xcount", "?")
        log.info("Geogrid 完成。输出格网: %s × %s", ycount, xcount)
    finally:
        os.chdir(orig_dir)

    _ensure_scale_factor(working_dir, geogrid_run_info)
    return geogrid_run_info



def _ensure_scale_factor(working_dir: str, geogrid_run_info: dict) -> None:
    """若 window_scale_factor.tif 不存在则创建全1占位栅格。"""
    sf_path  = Path(working_dir) / "window_scale_factor.tif"
    loc_path = Path(working_dir) / "window_location.tif"

    if sf_path.exists() or not loc_path.exists():
        return

    arr, gt, proj = read_geotiff(str(loc_path))
    if arr.ndim == 3:
        ycount, xcount = arr.shape[1], arr.shape[2]
    else:
        ycount, xcount = arr.shape

    ones = np.ones((ycount, xcount), dtype=np.float64)
    write_geotiff(str(sf_path), [ones, ones], gt, proj,
                  band_names=["scale_x", "scale_y"])
    log.info("已生成 window_scale_factor.tif（全1占位）")
