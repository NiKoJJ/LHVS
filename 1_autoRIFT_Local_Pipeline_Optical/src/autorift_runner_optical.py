"""
autorift_runner_optical.py — 光学影像 autoRIFT 特征追踪 + 速度场输出

与 S1 版本（autorift_runner.py）的核心差异：
  ┌─────────────────────────────────────────────────────────────────┐
  │  参数           │ SAR 版本               │ 光学版本              │
  ├─────────────────┼────────────────────────┼───────────────────────┤
  │ 影像加载        │ loadProduct() float32  │ loadProductOptical()  │
  │                 │ 无地理参考              │ 有 GeoTransform       │
  │ Dy0 符号        │ Dy0 = -Dy0（方位向）   │ 不取反（地图坐标）     │
  │ 预处理滤波      │ WAL / HPS / SOB        │ HPS（S2/L8/L9）       │
  │                 │                        │ 外部 Wallis/FFT 已完成 │
  │ OverSampleRatio │ {chip: 32/64/128/128}  │ {chip: 16/32/64/64}   │
  │ 速度来源        │ offset2vx/vy 系数转换  │ 相同（Geogrid 输出）   │
  └─────────────────┴────────────────────────┴───────────────────────┘

输出文件：
  velocity_{pair}.tif   → Band1=Vx  Band2=Vy  Band3=|V|  (m/yr, float32)
  offset_{pair}.tif     → Band1=DX  Band2=DY  Band3=InterpMask  Band4=ChipSizeX
  velocity_{pair}.nc    → CF-1.8 NetCDF（含 CRS / 误差 / 元数据）
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal, osr

from .utils import ensure_dir, read_geotiff, write_geotiff

gdal.UseExceptions()
log = logging.getLogger(__name__)

NODATA_VEL    = -32767.0
NODATA_OFFSET = -32767.0


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run_autorift_optical(
    ref_path: str,
    sec_path: str,
    meta_r,
    meta_s,
    window_dir: str,
    output_dir: str,
    pair_name: str,
    platform: str = "S2",
    oversample_ratio: int = 64,
    n_threads: int = 4,
    save_offsets: bool = True,
    velocity_max_m_yr: float = 20000.0,
    vx0_tif: Optional[str] = None,
    vy0_tif: Optional[str] = None,
    save_netcdf: bool = False,
    date_ref: str = "",
    date_sec: str = "",
    epsg: int = 3031,
    zero_mask_ref: Optional[np.ndarray] = None,
    zero_mask_sec: Optional[np.ndarray] = None,
    geogrid_run_info: Optional[dict] = None,   # ← 新增
) -> dict:
    """运行光学 autoRIFT 并保存速度场。

    Parameters
    ----------
    ref_path, sec_path : 经预处理的波段文件路径
    meta_r, meta_s     : load_optical_metadata_pair() 返回的元数据对象
    platform           : 影响预处理滤波策略（"S2"/"L8"/"L9" 使用 HPS；
                         "L7"/"L4"/"L5" 外部已完成，此处设为 "hps_skip"）
    zero_mask_ref/sec  : L7 外部滤波输出的零值掩膜（可选）
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    ensure_dir(output_dir)

    try:
        from autoRIFT import autoRIFT
    except ImportError:
        raise ImportError("请安装 autoRIFT：conda install -c conda-forge autorift")

    # ── 1. 加载影像（使用 meta 中的配准窗口）────────────────────────────────
    log.info("加载配准影像对...")
    I1, I2 = _load_optical_image_pair(ref_path, sec_path, meta_r, meta_s)
    log.info("参考景尺寸: %s | 次景尺寸: %s", I1.shape, I2.shape)

    # ── 2. 加载 Geogrid 输出 window_*.tif ───────────────────────────────────
    wd = Path(window_dir)
    log.info("加载 window_*.tif (%s)...", window_dir)

    xGrid, yGrid, geo_transform, projection = _load_grid_location(
        str(wd / "window_location.tif")
    )
    Dx0,     Dy0     = _load_2band(str(wd / "window_offset.tif"))
    SRx0,    SRy0    = _load_2band(str(wd / "window_search_range.tif"))
    CSMINx0, CSMINy0 = _load_2band_optional(str(wd / "window_chip_size_min.tif"))
    CSMAXx0, CSMAXy0 = _load_2band_optional(str(wd / "window_chip_size_max.tif"))
    ssm_mask          = _load_ssm_mask(str(wd / "window_stable_surface_mask.tif"), xGrid.shape)

    offset2vx_path = str(wd / "window_rdr_off2vel_x_vec.tif")
    offset2vy_path = str(wd / "window_rdr_off2vel_y_vec.tif")
    NODATA_VAL     = -32767


    # ── 3. 图像 noData 掩膜 ──────────────────────────────────────────────────
    log.info("构建 noDataMask...")
    noDataMask, zeroMask = _build_image_nodata_mask_optical(
        I1, I2, xGrid, yGrid, ssm_mask, NODATA_VAL
    )

    # ── 3.5 Pad 所有格网到 32 的整数倍 ──────────────────────────────────────
    # 必须在 noDataMask 构建完成之后执行
    GRID_MULTIPLE = 32
    orig_shape = xGrid.shape
    log.info("格网原始尺寸: %s → pad 至 32 整数倍", orig_shape)

    def _pad(arr, fill=0):
        if arr is None:
            return None
        r, c = arr.shape
        r_new = int(np.ceil(r / GRID_MULTIPLE) * GRID_MULTIPLE)
        c_new = int(np.ceil(c / GRID_MULTIPLE) * GRID_MULTIPLE)
        if r_new == r and c_new == c:
            return arr
        out = np.full((r_new, c_new), fill, dtype=arr.dtype)
        out[:r, :c] = arr
        return out

    xGrid      = _pad(xGrid,    fill=0)
    yGrid      = _pad(yGrid,    fill=0)
    noDataMask = _pad(noDataMask.astype(np.int32), fill=1).astype(bool)
    Dx0        = _pad(Dx0,      fill=0)
    Dy0        = _pad(Dy0,      fill=0)
    if SRx0 is not None:
        SRx0 = _pad(SRx0, fill=0)
        SRy0 = _pad(SRy0, fill=0)
    if CSMINx0 is not None:
        CSMINx0 = _pad(CSMINx0, fill=0)
        CSMINy0 = _pad(CSMINy0, fill=0)
        CSMAXx0 = _pad(CSMAXx0, fill=0)
        CSMAXy0 = _pad(CSMAXy0, fill=0)
    ssm_mask   = _pad(ssm_mask.astype(np.int32), fill=0).astype(bool)

    padded_shape = xGrid.shape
    log.info("Pad 后格网尺寸: %s", padded_shape)


    # ── 4. 初始化 autoRIFT ──────────────────────────────────────────────────
    log.info("初始化 autoRIFT 对象（光学模式）...")
    obj = autoRIFT()
    obj.MultiThread = 0
    obj.I1 = I1
    obj.I2 = I2
    obj.xGrid = xGrid.copy()
    obj.yGrid = yGrid.copy()

    # 搜索范围
    if SRx0 is not None:
        obj.SearchLimitX = SRx0.copy()
        obj.SearchLimitY = SRy0.copy()
        mask_nz = (obj.SearchLimitX != 0) & (obj.SearchLimitX != NODATA_VAL)
        obj.SearchLimitX[mask_nz] += 2
        obj.SearchLimitY[mask_nz] += 2
    else:
        obj.SearchLimitX = 15
        obj.SearchLimitY = 15

    # Chip size（光学默认更小：16/32px）
    if CSMINx0 is not None:
        # ── 从 geogrid_run_info 还原 ChipSize0X（与 testautoRIFT.py 逻辑完全对齐）
        if geogrid_run_info is not None:
            chipsizex0   = geogrid_run_info.get("chipsizex0",   240.0)
            gridspacingx = geogrid_run_info.get("gridspacingx", 240.0)
            pixsizex     = geogrid_run_info.get("XPixelSize",    30.0)
        else:
            # 回退：从 csmin 数组中估算
            valid = (CSMINx0 > 0) & (CSMINx0 != NODATA_VAL)
            pixsizex     = float(np.nanmin(CSMINx0[valid])) if valid.any() else 30.0
            chipsizex0   = 240.0
            gridspacingx = 240.0

        # 与官方 testautoRIFT.runAutorift() 完全一致的计算公式
        obj.ChipSize0X   = int(np.ceil(chipsizex0 / pixsizex / 4) * 4)
        obj.GridSpacingX = int(obj.ChipSize0X * gridspacingx / chipsizex0)
        log.info(
            "ChipSize0X=%d px  GridSpacingX=%d px  (chipsizex0=%.0f m, pixsize=%.1f m)",
            obj.ChipSize0X, obj.GridSpacingX, chipsizex0, pixsizex,
        )

        # ScaleChipSizeY
        valid = (CSMINx0 != NODATA_VAL) & (CSMINy0 != NODATA_VAL) & (CSMINx0 > 0)
        obj.ScaleChipSizeY = float(np.median(CSMINy0[valid] / CSMINx0[valid])) \
                             if valid.any() else 1.0

        # 取整为 ChipSize0X 的整数倍（关键修复）
        def _snap_to_chip0(arr: np.ndarray, chip0: int, nodata: float) -> np.ndarray:
            """将像素数取整为 ChipSize0X 的整数倍（autoRIFT 硬性要求）。"""
            out = np.where(
                (arr == nodata) | (arr <= 0),
                0,
                np.maximum(chip0, np.round(arr / chip0).astype(np.int32) * chip0),
            ).astype(np.int32)
            return out

        c0 = obj.ChipSize0X
        csmin_snapped = _snap_to_chip0(CSMINx0, c0, NODATA_VAL)
        csmax_snapped = _snap_to_chip0(CSMAXx0, c0, NODATA_VAL)

        # max 不得小于 min
        bad = (csmax_snapped > 0) & (csmin_snapped > 0) & (csmax_snapped < csmin_snapped)
        csmax_snapped[bad] = csmin_snapped[bad]

        obj.ChipSizeMinX = csmin_snapped
        obj.ChipSizeMaxX = csmax_snapped

        log.info(
            "ChipSize 取整后：min=[%d, %d]  max=[%d, %d]  (ChipSize0X=%d)",
            int(csmin_snapped[csmin_snapped > 0].min()) if (csmin_snapped > 0).any() else -1,
            int(csmin_snapped.max()),
            int(csmax_snapped[csmax_snapped > 0].min()) if (csmax_snapped > 0).any() else -1,
            int(csmax_snapped.max()),
            c0,
        )

    # 初始偏移量
    # 【关键差异 vs. SAR】光学在地图坐标，Dy0 不取反
    if Dx0 is not None:
        obj.Dx0 = Dx0.copy()
        obj.Dy0 = Dy0.copy()       # ← 光学：不取反（SAR 版本: Dy0 = -Dy0）
    else:
        obj.Dx0 = np.zeros_like(xGrid, dtype=np.float32)
        obj.Dy0 = np.zeros_like(yGrid, dtype=np.float32)

    # 应用 noDataMask
    obj.xGrid[noDataMask] = 0
    obj.yGrid[noDataMask] = 0
    obj.Dx0[noDataMask]   = 0
    obj.Dy0[noDataMask]   = 0
    if SRx0 is not None:
        obj.SearchLimitX[noDataMask] = 0
        obj.SearchLimitY[noDataMask] = 0
    if CSMINx0 is not None:
        obj.ChipSizeMinX[noDataMask] = 0
        obj.ChipSizeMaxX[noDataMask] = 0

    # L7 零值掩膜处理
    if zero_mask_ref is not None and zero_mask_sec is not None:
        combined_zero = zero_mask_ref | zero_mask_sec
        obj.zeroMask = combined_zero.astype(np.uint8)
        log.debug("L7 zeroMask: %d 像素", combined_zero.sum())

    # ── 5. 预处理滤波 ────────────────────────────────────────────────────────
    t0 = time.time()
    _apply_filter_optical(obj, platform)
    obj.uniform_data_type()
    log.info("预处理完成 (%.1f s)", time.time() - t0)

    # ── 6. OverSampleRatio（光学模式使用更小值）─────────────────────────────
    obj.sparseSearchSampleRate = 1
    if CSMINx0 is not None and hasattr(obj, "ChipSize0X"):
        c0 = obj.ChipSize0X
        # 官方光学 OverSampleRatio 字典（小于 SAR）
        obj.OverSampleRatio = {
            c0:     16,
            c0 * 2: 32,
            c0 * 4: 64,
            c0 * 8: 64,
        }
    else:
        obj.OverSampleRatio = oversample_ratio

    # ── 7. 运行 autoRIFT ─────────────────────────────────────────────────────
    t0 = time.time()
    log.info("运行 autoRIFT 特征追踪（光学模式）...")
    obj.runAutorift()
    log.info("autoRIFT 完成 (%.1f s)", time.time() - t0)

    # 扩张 noDataMask
    kernel     = np.ones((3, 3), np.uint8)
    noDataMask = cv2.dilate(noDataMask.astype(np.uint8), kernel, iterations=1).astype(bool)

    # ── 8. 整理结果 ──────────────────────────────────────────────────────────
    # ── 整理结果（裁回原始格网尺寸）─────────────────────────────────────────
    r0, c0 = orig_shape

    def _crop(arr):
        a = np.asarray(arr, dtype=np.float32)
        if a.shape != padded_shape:
            # autoRIFT 可能返回略小的数组，先 pad 再裁
            tmp = np.full(padded_shape, np.nan, dtype=np.float32)
            tmp[:a.shape[0], :a.shape[1]] = a
            a = tmp
        return a[:r0, :c0]

    DX         = _crop(obj.Dx)
    DY         = _crop(obj.Dy)
    INTERPMASK = _crop(obj.InterpMask)
    CHIPSIZEX  = _crop(obj.ChipSizeX)
    noDataMask = noDataMask[:r0, :c0]
    grid_shape = orig_shape  # 后续误差估计、输出使用原始尺寸

    # ── 9. 像素偏移 → 速度（m/yr）──────────────────────────────────────────
    if not Path(offset2vx_path).exists():
        raise FileNotFoundError(
            f"未找到速度转换系数: {offset2vx_path}\n请确认 Geogrid 正常完成。"
        )
    VX, VY = _offsets_to_velocity(DX, DY, offset2vx_path, offset2vy_path, NODATA_VEL)

    # 速度阈值过滤
    v_valid = np.isfinite(VX) & np.isfinite(VY)
    v_mag   = np.where(v_valid, np.sqrt(VX**2 + VY**2), np.nan)
    outlier = v_mag > velocity_max_m_yr
    if outlier.any():
        log.warning("速度阈值过滤: %d 像素超过 %.0f m/yr → NoData", outlier.sum(), velocity_max_m_yr)
        VX[outlier] = np.nan
        VY[outlier] = np.nan
        v_mag[outlier] = np.nan

    V = np.where(np.isfinite(VX) & np.isfinite(VY), np.sqrt(VX**2 + VY**2), np.nan)

    # ── 10. 误差估计 ─────────────────────────────────────────────────────────
    ssm_window = _load_ssm_mask_raw(str(wd / "window_stable_surface_mask.tif"), grid_shape)
    stable = ssm_window & np.isfinite(VX) & np.isfinite(VY)
    stable_count = int(stable.sum())

    if stable_count >= 5:
        if vx0_tif and Path(vx0_tif).exists():
            vx0_arr, _, _ = read_geotiff(vx0_tif)
            vy0_arr, _, _ = read_geotiff(vy0_tif)
            vx0_arr = vx0_arr.astype(np.float32)
            vy0_arr = vy0_arr.astype(np.float32)
            vx0_arr[vx0_arr == NODATA_VEL] = np.nan
            vy0_arr[vy0_arr == NODATA_VEL] = np.nan
            # 重采样到 grid 尺寸
            vx0_grid = _resample_array(vx0_arr, grid_shape)
            vy0_grid = _resample_array(vy0_arr, grid_shape)
            valid_stable = stable & np.isfinite(vx0_grid) & np.isfinite(vy0_grid)
            vx_error = float(np.nanstd(VX[valid_stable] - vx0_grid[valid_stable])) if valid_stable.any() else np.nan
            vy_error = float(np.nanstd(VY[valid_stable] - vy0_grid[valid_stable])) if valid_stable.any() else np.nan
            error_method = "std_residual_vs_prior"
        else:
            vx_error = float(np.nanstd(VX[stable]))
            vy_error = float(np.nanstd(VY[stable]))
            error_method = "std_stable_surface"
    else:
        log.warning("稳定地表像素不足（%d），无法估算误差。", stable_count)
        vx_error = vy_error = np.nan
        error_method = "insufficient_stable_pixels"

    v_error = float(np.sqrt(vx_error**2 + vy_error**2)) if np.isfinite(vx_error) else np.nan
    log.info(
        "误差估计 (%s): Vx_err=%.2f  Vy_err=%.2f  V_err=%.2f  stable=%d",
        error_method, vx_error or 0, vy_error or 0, v_error or 0, stable_count,
    )

    # ── 11. 写出 GeoTIFF ─────────────────────────────────────────────────────
    vel_path = str(Path(output_dir) / f"velocity_{pair_name}.tif")
    _save_velocity(vel_path, VX, VY, V, geo_transform, projection)

    results = {
        "velocity_tif": vel_path,
        "vx_error":     vx_error,
        "vy_error":     vy_error,
        "v_error":      v_error,
        "stable_count": stable_count,
        "error_method": error_method,
    }

    if save_offsets:
        off_path = str(Path(output_dir) / f"offset_{pair_name}.tif")
        _save_offsets(off_path, DX, DY, INTERPMASK, CHIPSIZEX, geo_transform, projection)
        results["offset_tif"] = off_path

    if save_netcdf:
        nc_path = str(Path(output_dir) / f"velocity_{pair_name}.nc")
        _save_netcdf(
            nc_path, VX, VY, V, CHIPSIZEX, INTERPMASK,
            geo_transform, projection, epsg,
            pair_name, date_ref, date_sec,
            stable_count, vx_error, vy_error, v_error, error_method,
        )
        results["netcdf"] = nc_path

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 预处理滤波（光学模式）
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filter_optical(obj, platform: str) -> None:
    """光学平台专属预处理滤波。

    S2 / L8 / L9 → HPS（高通）
    L7 / L4 / L5 → 外部已完成（Wallis-fill 或 FFT），此处仅做统一数据类型
    """
    platform = platform.upper()
    t0 = time.time()
    if platform in ("S2", "L8", "L9"):
        log.info("预处理: HPS 高通滤波（%s）...", platform)
        obj.preprocess_filt_hps()
    elif platform in ("L7", "L4", "L5"):
        log.info("预处理: 外部滤波已完成（%s），跳过内部滤波", platform)
        # 不调用任何内部滤波方法
    else:
        log.warning("未知平台 '%s'，默认使用 HPS", platform)
        obj.preprocess_filt_hps()
    log.info("预处理完成 (%.1f s)", time.time() - t0)


# ─────────────────────────────────────────────────────────────────────────────
# 影像加载（使用配准窗口）
# ─────────────────────────────────────────────────────────────────────────────

def _load_optical_image_pair(
    ref_path: str,
    sec_path: str,
    meta_r,
    meta_s,
) -> Tuple[np.ndarray, np.ndarray]:
    """从配准元数据中读取公共区域影像。"""
    ds1 = gdal.Open(ref_path, gdal.GA_ReadOnly)
    I1  = ds1.ReadAsArray(
        xoff=meta_r._x1a, yoff=meta_r._y1a,
        xsize=meta_r._xsize, ysize=meta_r._ysize,
    ).astype(np.float32)
    ds1 = None

    ds2 = gdal.Open(sec_path, gdal.GA_ReadOnly)
    I2  = ds2.ReadAsArray(
        xoff=meta_s._x1a, yoff=meta_s._y1a,
        xsize=meta_s._xsize, ysize=meta_s._ysize,
    ).astype(np.float32)
    ds2 = None
    return I1, I2


# ─────────────────────────────────────────────────────────────────────────────
# noData 掩膜（光学版本：零值即无效，不适用于 L7 SLC-off）
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_nodata_mask_optical(
    I1: np.ndarray,
    I2: np.ndarray,
    xGrid: np.ndarray,
    yGrid: np.ndarray,
    init_mask: np.ndarray,
    nodata_val: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建 noDataMask（基于图像零值）和 zeroMask（全图零值位置）。

    【注意】L7 SLC-off 不应依赖零值判断（条纹本身为零），
    其 zero_mask 应从外部 Wallis-fill 滤波结果传入，见 run_autorift_optical 参数。
    此处统一处理；调用方在 L7 情况下覆盖 obj.zeroMask。
    """
    mask     = init_mask.copy()
    rows, cols = xGrid.shape
    h1, w1   = I1.shape
    zeroMask = (I1 == 0) | (I2 == 0)

    for ii in range(rows):
        for jj in range(cols):
            xi, yi = int(xGrid[ii, jj]), int(yGrid[ii, jj])
            if xi == nodata_val or yi == nodata_val or xi < 1 or yi < 1:
                mask[ii, jj] = True
                continue
            r, c = yi - 1, xi - 1
            if r >= h1 or c >= w1 or r < 0 or c < 0:
                mask[ii, jj] = True
                continue
            try:
                if I1[r, c] == 0 or I2[r, c] == 0:
                    mask[ii, jj] = True
            except IndexError:
                mask[ii, jj] = True

    return mask, zeroMask


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_even_int(arr: np.ndarray, nodata: float, min_val: int = 4) -> np.ndarray:
    return np.where(
        (arr == nodata) | (arr <= 0),
        0,
        np.maximum(min_val, (np.round(arr / 2) * 2)).astype(np.int32),
    ).astype(np.int32)


def _load_grid_location(path: str):
    arr, gt, proj = read_geotiff(path)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"window_location.tif 应为 2 波段，实际 shape={arr.shape}")
    return arr[0].astype(np.int32), arr[1].astype(np.int32), gt, proj


def _load_2band(path: str):
    arr, _, _ = read_geotiff(path)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return arr[0].astype(np.float32), arr[1].astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32), arr.astype(np.float32)
    return None, None


def _load_2band_optional(path: str):
    if not Path(path).exists():
        return None, None
    return _load_2band(path)


def _load_ssm_mask(ssm_path: str, shape: tuple) -> np.ndarray:
    if not Path(ssm_path).exists():
        return np.zeros(shape, dtype=bool)
    arr, _, _ = read_geotiff(ssm_path)
    mask = (arr != 0).astype(bool)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != shape:
        mask = mask[:shape[0], :shape[1]]
    return mask


def _load_ssm_mask_raw(ssm_path: str, shape: tuple) -> np.ndarray:
    return _load_ssm_mask(ssm_path, shape)


def _pad_to(arr: np.ndarray, shape: tuple) -> np.ndarray:
    if arr.shape == shape:
        return arr
    out = np.full(shape, np.nan, dtype=np.float32)
    r = min(arr.shape[0], shape[0])
    c = min(arr.shape[1], shape[1])
    out[:r, :c] = arr[:r, :c]
    return out


def _offsets_to_velocity(DX, DY, vx_path, vy_path, nodata_val):
    def _load_coeff(path):
        arr, _, _ = read_geotiff(path)
        if arr.ndim == 3:
            b1, b2 = arr[0].astype(np.float64), arr[1].astype(np.float64)
        else:
            b1 = arr.astype(np.float64)
            b2 = np.zeros_like(b1)
        b1[b1 == nodata_val] = np.nan
        b2[b2 == nodata_val] = np.nan
        return b1, b2

    vx1, vx2 = _load_coeff(vx_path)
    vy1, vy2 = _load_coeff(vy_path)
    DX64, DY64 = DX.astype(np.float64), DY.astype(np.float64)
    VX = (vx1 * DX64 + vx2 * DY64).astype(np.float32)
    VY = (vy1 * DX64 + vy2 * DY64).astype(np.float32)
    VX[~np.isfinite(DX64)] = np.nan
    VY[~np.isfinite(DY64)] = np.nan
    return VX, VY


def _resample_array(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """简单最近邻重采样（用于误差估计时对齐格网）。"""
    if arr.shape == target_shape:
        return arr
    rows, cols = target_shape
    sr = arr.shape[0] / rows
    sc = arr.shape[1] / cols
    ri = np.clip((np.arange(rows) * sr).astype(int), 0, arr.shape[0] - 1)
    ci = np.clip((np.arange(cols) * sc).astype(int), 0, arr.shape[1] - 1)
    return arr[ri[:, None], ci[None, :]]


def _save_velocity(path, VX, VY, V, gt, proj):
    def _fill(a):
        x = a.copy(); x[~np.isfinite(x)] = NODATA_VEL; return x.astype(np.float32)
    write_geotiff(path, [_fill(VX), _fill(VY), _fill(V)], gt, proj,
                  nodata=NODATA_VEL, band_names=["Vx_m_yr", "Vy_m_yr", "V_m_yr"])
    log.info("速度场已写入: %s", path)


def _save_offsets(path, DX, DY, INTERPMASK, CHIPSIZEX, gt, proj):
    def _fill(a, nv):
        x = a.copy(); x[~np.isfinite(x)] = nv; return x.astype(np.float32)
    write_geotiff(path,
                  [_fill(DX, NODATA_OFFSET), _fill(DY, NODATA_OFFSET),
                   _fill(INTERPMASK, 0), _fill(CHIPSIZEX, 0)],
                  gt, proj, nodata=NODATA_OFFSET,
                  band_names=["DX_px", "DY_px", "InterpMask", "ChipSizeX"])
    log.info("偏移量已写入: %s", path)


def _save_netcdf(
    path, VX, VY, V, CHIPSIZEX, INTERPMASK,
    geo_transform, projection, epsg,
    pair_name, date_ref, date_sec,
    stable_count, vx_error, vy_error, v_error, error_method,
):
    """写出 CF-1.8 NetCDF（与 S1 版本格式完全对齐）。"""
    try:
        import xarray as xr
    except ImportError:
        log.warning("xarray 未安装，跳过 NetCDF 输出。pip install xarray")
        return

    rows, cols = VX.shape
    x0, dx = geo_transform[0], geo_transform[1]
    y0, dy = geo_transform[3], geo_transform[5]
    x_coords = x0 + dx * (np.arange(cols) + 0.5)
    y_coords  = y0 + dy * (np.arange(rows) + 0.5)

    srs = osr.SpatialReference(projection)
    crs_wkt = srs.ExportToWkt()

    def _fill(a):
        x = a.copy(); x[~np.isfinite(x)] = NODATA_VEL; return x.astype(np.float32)

    # 时间差
    try:
        dt_days = (
            datetime.strptime(date_sec, "%Y%m%d") - datetime.strptime(date_ref, "%Y%m%d")
        ).days
    except Exception:
        dt_days = 0

    ds = xr.Dataset(
        {
            "vx": xr.Variable(["y", "x"], _fill(VX),
                attrs={"long_name": "x-component of ice velocity", "units": "m/yr",
                       "_FillValue": float(NODATA_VEL), "grid_mapping": "crs"}),
            "vy": xr.Variable(["y", "x"], _fill(VY),
                attrs={"long_name": "y-component of ice velocity", "units": "m/yr",
                       "_FillValue": float(NODATA_VEL), "grid_mapping": "crs"}),
            "v":  xr.Variable(["y", "x"], _fill(V),
                attrs={"long_name": "ice speed", "units": "m/yr",
                       "_FillValue": float(NODATA_VEL), "grid_mapping": "crs"}),
            "chip_size_max": xr.Variable(["y", "x"], _fill(CHIPSIZEX).astype(np.float32),
                attrs={"units": "m", "long_name": "maximum chip size used in autoRIFT",
                       "_FillValue": 0.0, "grid_mapping": "crs"}),
            "interp_mask": xr.Variable(["y", "x"], INTERPMASK.astype(np.uint8),
                attrs={"long_name": "valid pixel flag", "_FillValue": np.uint8(0),
                       "grid_mapping": "crs"}),
            "crs": xr.Variable([], np.int32(epsg),
                attrs={"grid_mapping_name": "projection", "crs_wkt": crs_wkt,
                       "epsg_code": f"EPSG:{epsg}"}),
        },
        coords={
            "x": xr.Variable("x", x_coords.astype(np.float64),
                attrs={"standard_name": "projection_x_coordinate", "units": "m"}),
            "y": xr.Variable("y", y_coords.astype(np.float64),
                attrs={"standard_name": "projection_y_coordinate", "units": "m"}),
        },
        attrs={
            "Conventions":    "CF-1.8",
            "title":          f"Optical ice velocity: {pair_name}",
            "date_created":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "institution":    "autoRIFT_Optical_Local_Pipeline",
            "pair_name":      pair_name,
            "reference_date": date_ref,
            "secondary_date": date_sec,
            "dt_days":        dt_days,
            "EPSG":           epsg,
            "stable_count":   stable_count,
            "error_method":   error_method,
            "vx_error":       round(float(vx_error), 3) if np.isfinite(vx_error) else float(NODATA_VEL),
            "vy_error":       round(float(vy_error), 3) if np.isfinite(vy_error) else float(NODATA_VEL),
            "v_error":        round(float(v_error),  3) if np.isfinite(v_error)  else float(NODATA_VEL),
        },
    )

    encoding = {var: {"zlib": True, "complevel": 4, "shuffle": True}
                for var in ["vx", "vy", "v", "chip_size_max", "interp_mask"]}

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding, format="NETCDF4")
    log.info("NetCDF 已写入: %s", path)

def _pad_to_multiple(arr: np.ndarray, multiple: int = 32, fill=0) -> Tuple[np.ndarray, tuple]:
    """将二维数组 pad 到 multiple 的整数倍，返回 (padded_arr, original_shape)。"""
    r, c = arr.shape
    r_new = int(np.ceil(r / multiple) * multiple)
    c_new = int(np.ceil(c / multiple) * multiple)
    if r_new == r and c_new == c:
        return arr, (r, c)
    out = np.full((r_new, c_new), fill, dtype=arr.dtype)
    out[:r, :c] = arr
    return out, (r, c)