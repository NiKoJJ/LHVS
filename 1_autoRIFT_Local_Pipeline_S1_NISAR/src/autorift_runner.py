"""
autorift_runner.py — 直接驱动 autoRIFT 并输出速度场 GeoTIFF / NetCDF

【v3 新增】
  - 速度阈值过滤（velocity_max_m_yr，从 config 读取）
  - Vx_Error / Vy_Error 误差估计（两种模式）
      有 vx0/vy0：误差 = std(Vx - vx0) over 稳定地表像素
      无 vx0/vy0：误差 = std(Vx)       over 稳定地表像素（真实速度视为 0）
  - CF-1.8 NetCDF 输出（xarray，含 CRS / 元数据 / 压缩）

【v2 保留修复】
  - _build_image_nodata_mask 返回 (noDataMask, zeroMask) 双掩膜
  - obj.zeroMask 供 Wallis 滤波使用
  - 边界检查防止 OpenCV 崩溃

输出文件：
  velocity_{pair}.tif   → Band1=Vx  Band2=Vy  Band3=|V|  (m/yr, float32)
  offset_{pair}.tif     → Band1=DX  Band2=DY  Band3=InterpMask  Band4=ChipSizeX  (pixel)
  velocity_{pair}.nc    → CF-1.8 NetCDF（若 save_netcdf=True）
"""

import logging
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal, osr

from .utils import read_geotiff, write_geotiff, ensure_dir

gdal.UseExceptions()
log = logging.getLogger(__name__)

NODATA_VEL    = -32767.0
NODATA_OFFSET = -32767.0


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_autorift(
    ref_tif: str,
    sec_tif: str,
    window_dir: str,
    output_dir: str,
    pair_name: str,
    filter_type: Optional[str] = None, 
    filter_width: Optional[int] = None, 
    oversample_ratio: int = 64,
    n_threads: int = 4,
    save_offsets: bool = True,
    # ── v3 新增参数 ──────────────────────────────────────────
    velocity_max_m_yr: float = 20000.0,
    vx0_tif: Optional[str] = None,
    vy0_tif: Optional[str] = None,
    save_netcdf: bool = False,
    date_ref: str = "",
    date_sec: str = "",
    epsg: int = 3031,
    # ── v4 新增：Geogrid 运行信息（ChipSize0X / GridSpacingX 计算必需）────
    geogrid_run_info: Optional[dict] = None,
) -> dict:
    # config 中 filter_type/filter_width 可能为 null → 统一回退
    # filter_type  = filter_type  or "WAL"
    # filter_width = filter_width or 21
    """运行 autoRIFT 并保存速度场 GeoTIFF（及可选 NetCDF）。

    Parameters
    ----------
    velocity_max_m_yr : float
        物理异常值过滤阈值（m/yr）。超出此值的像素置为 NoData。
    vx0_tif / vy0_tif : str, optional
        参考速度栅格（通常为 SPS_0120m_vx0.tif 重采样版本）。
        提供时：Vx_Error = std(Vx - vx0) over 稳定地表；
        不提供：Vx_Error = std(Vx)        over 稳定地表（稳定地表视为零速）。
    save_netcdf : bool
        是否同时写出 CF-1.8 NetCDF（xarray）。
    date_ref / date_sec : str
        YYYYMMDD 格式日期，写入 NetCDF 全局属性。
    epsg : int
        投影 EPSG 代码，写入 NetCDF CRS 信息。
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    ensure_dir(output_dir)

    try:
        from autoRIFT import autoRIFT
    except ImportError:
        raise ImportError(
            "请安装 autoRIFT：conda install -c conda-forge autorift\n"
            "或参考 README 中的安装说明。"
        )

    # ── 加载影像 ─────────────────────────────────────────────────────────────
    log.info("加载幅度图...")
    I1 = _load_image(ref_tif)
    I2 = _load_image(sec_tif)
    log.info("参考景尺寸: %s | 次景尺寸: %s", I1.shape, I2.shape)

    # ── 加载 Geogrid window_*.tif ────────────────────────────────────────────
    wd = Path(window_dir)
    log.info("加载 window_*.tif (%s)...", window_dir)

    xGrid, yGrid, geo_transform, projection = _load_grid_location(
        str(wd / "window_location.tif")
    )
    Dx0,    Dy0    = _load_2band(str(wd / "window_offset.tif"))
    SRx0,   SRy0   = _load_2band(str(wd / "window_search_range.tif"))
    CSMINx0, CSMINy0 = _load_2band_optional(str(wd / "window_chip_size_min.tif"))
    CSMAXx0, CSMAXy0 = _load_2band_optional(str(wd / "window_chip_size_max.tif"))
    # Stable-surface pixels are valid matching pixels.  They are used later for
    # error statistics and must not be added to the matching no-data mask.
    noDataMask = np.zeros(xGrid.shape, dtype=bool)

    offset2vx_path = str(wd / "window_rdr_off2vel_x_vec.tif")
    offset2vy_path = str(wd / "window_rdr_off2vel_y_vec.tif")
    NODATA_VAL     = -32767

    # ── 构建图像 no-data 掩膜 ────────────────────────────────────────────────
    log.info("构建 noDataMask（窗口越界与 SAR 零幅度区域）...")
    # NISAR amplitude extraction already filters and normalizes the images.  In
    # that path zeroMask is never consumed, so do not allocate another full
    # image-sized boolean array (several GB for a 30 m NISAR pair).
    keep_zero_mask = bool(filter_type)
    noDataMask, zeroMask = _build_image_nodata_mask(
        I1, I2, xGrid, yGrid, noDataMask, NODATA_VAL,
        build_zero_mask=keep_zero_mask,
    )

    # ── 初始化 autoRIFT ──────────────────────────────────────────────────────
    log.info("初始化 autoRIFT 对象...")
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

    # Chip size
    # ──────────────────────────────────────────────────────────────────────────
    # 官方 testautoRIFT.py 逻辑（逐行对齐）：
    #
    #   obj.ChipSize0X = int(np.ceil(chipsizex0 / pixsizex / 4) * 4)
    #   obj.GridSpacingX = int(obj.ChipSize0X * gridspacingx / chipsizex0)
    #
    # 所有 ChipSizeMin/Max 必须是 ChipSize0X 的整数倍，而非 4 的倍数。
    # ChipSize0X 由 Geogrid 输出的 chipsizex0（m）和 XPixelSize（m/px）决定。
    # 示例：chipsizex0=240m, pixsizex=2.529m → ChipSize0X=96px
    #        → 所有 chip size 必须是 96 的倍数（96, 192, 288, ...）
    # ──────────────────────────────────────────────────────────────────────────
    if CSMINx0 is not None:
        # ── 从 geogrid_run_info 提取 Geogrid 参数 ─────────────────────────
        if geogrid_run_info is not None:
            chipsizex0   = float(geogrid_run_info.get('chipsizex0',   240.0))
            pixsizex     = float(geogrid_run_info.get('XPixelSize',     1.0))
            gridspacingx = float(geogrid_run_info.get('gridspacingx', 120.0))
            log.info("精确加载 Geogrid 参数: chipsizex0=%.1f m, pixsizex=%.4f m/px", chipsizex0, pixsizex)
        else:
            # geogrid_run_info 未传入时的保守默认（与旧版行为一致）
            log.warning(
                "geogrid_run_info 未提供，无法精确计算 ChipSize0X；"
                "使用回退值 chipsizex0=240, pixsizex=1.0, gridspacingx=120。"
                "建议在调用 run_autorift() 时传入 geogrid_run_info。"
            )
            chipsizex0, pixsizex, gridspacingx = 240.0, 1.0, 120.0

        # ── 计算可兼容的 ChipSize0X / GridSpacingX ────────────────────────
        #
        # autoRIFT requires ChipSize0X % GridSpacingX == 0 whenever the
        # grid spacing is smaller than the base chip.  The upstream formula
        # truncates both quantities independently, which fails for a 30 m DEM
        # and NISAR's 4.7386 m ground-range pixels:
        #
        #   ChipSize0X=52 px, GridSpacingX=6 px, 52 % 6 != 0
        #
        # Keep the requested Geogrid spacing, but select the nearest
        # four-pixel-aligned base chip that is a multiple of that spacing.
        # This changes the internal base chip by at most one compatibility
        # unit; it does not change the Geogrid output nodes or their 30 m
        # GeoTransform.
        chip0, grid_spacing_px, requested_chip0, requested_grid_px = (
            _compatible_chip_grid(
                chipsizex0=chipsizex0,
                pixel_size_x=pixsizex,
                grid_spacing_m=gridspacingx,
            )
        )
        obj.ChipSize0X   = chip0
        obj.GridSpacingX = grid_spacing_px
        log.info(
            "ChipSize0X=%d px  GridSpacingX=%d px  "
            "(目标格网≈%.2f m；chipsizex0=%.1f m, pixsizex=%.4f m/px)",
            chip0, obj.GridSpacingX, obj.GridSpacingX * pixsizex,
            chipsizex0, pixsizex,
        )
        if (chip0, grid_spacing_px) != (requested_chip0, requested_grid_px):
            log.info(
                "为满足 autoRIFT 整数约束，基础 Chip 从 %d px 调整为 %d px；"
                "格网步长保持 %d px。",
                requested_chip0, chip0, grid_spacing_px,
            )

        # ── ScaleChipSizeY（用原始浮点值，取整前）──────────────────────────
        valid = (CSMINx0 != NODATA_VAL) & (CSMINy0 != NODATA_VAL) & (CSMINx0 > 0)
        obj.ScaleChipSizeY = float(np.median(CSMINy0[valid] / CSMINx0[valid])) if valid.any() else 1.0

        # ── 对齐到 autoRIFT 支持的 nested-search 层级 ──────────────────────
        # autorift() only evaluates ChipSize0X*[1, 2, 4, 8, ...].  Snapping
        # directly to arbitrary 3x/5x multiples makes the requested maximum
        # silently unreachable.  Keep the documented, tested 1/2/4/8 levels.
        csmin_aligned = _snap_chip_sizes(CSMINx0, chip0, NODATA_VAL)
        csmax_aligned = _snap_chip_sizes(CSMAXx0, chip0, NODATA_VAL)

        # max 不得小于 min
        bad = (csmax_aligned > 0) & (csmin_aligned > 0) & (csmax_aligned < csmin_aligned)
        csmax_aligned[bad] = csmin_aligned[bad]

        obj.ChipSizeMinX = csmin_aligned
        obj.ChipSizeMaxX = csmax_aligned

        log.debug(
            "ChipSizeMinX: valid range=[%d, %d]  ChipSizeMaxX: valid range=[%d, %d]  (单位: px)",
            int(csmin_aligned[csmin_aligned > 0].min()) if (csmin_aligned > 0).any() else -1,
            int(csmin_aligned.max()),
            int(csmax_aligned[csmax_aligned > 0].min()) if (csmax_aligned > 0).any() else -1,
            int(csmax_aligned.max()),
        )

        # ── OverSampleRatio：SAR 路径使用 ChipSize0X 分档字典（官方逻辑）─
        obj.OverSampleRatio = {
            chip0:     32,
            chip0 * 2: 64,
            chip0 * 4: 128,
            chip0 * 8: 128,
        }
        log.debug("OverSampleRatio（ChipSize0X 分档）: %s", obj.OverSampleRatio)

    else:
        obj.ChipSizeMinX = 16
        obj.ChipSizeMaxX = 64

    # 初始偏移量（SAR 方位向取反）
    if Dx0 is not None:
        obj.Dx0 = Dx0.copy()
        obj.Dy0 = -1.0 * Dy0.copy()
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

    obj.zeroMask = zeroMask.astype(np.uint8) if zeroMask is not None else None
    if zeroMask is not None:
        log.debug("zeroMask: %d / %d 像素标记为零值", zeroMask.sum(), zeroMask.size)

    # ── 预处理滤波 ───────────────────────────────────────────────────────────
    if filter_type:
        t0 = time.time()
        log.info( "预处理滤波: %s (width=%d)... ", filter_type, filter_width)
        _apply_filter(obj, filter_type, filter_width)
        obj.uniform_data_type()
        log.info( "预处理完成 (%.1f s) ", time.time() - t0)
    else:
        log.info( "跳过预处理滤波（数据已完成前置 HPS 滤波与 uint8 归一化）")

    obj.sparseSearchSampleRate = 1
    # OverSampleRatio：若 CSMINx0 存在，在上方 chip size 块中已设为 ChipSize0X 分档字典；
    # 若未设置（CSMINx0=None），此处使用标量回退
    if not isinstance(obj.OverSampleRatio, dict):
        obj.OverSampleRatio = oversample_ratio

    # Nested search requires each grid dimension to be an integer multiple of
    # the largest enabled chip scale, not merely an even number.
    _grid_shape_orig = xGrid.shape
    max_scale = _largest_chip_scale(obj.ChipSizeMaxX, obj.ChipSize0X)
    padded_shape = _shape_multiple(_grid_shape_orig, max_scale)
    if padded_shape != _grid_shape_orig:
        log.info(
            "格网尺寸 %s → %s（对齐到最大 nested-search 层级 %dx）",
            _grid_shape_orig, padded_shape, max_scale,
        )
        obj.xGrid = _pad_grid(obj.xGrid, padded_shape, fill=0)
        obj.yGrid = _pad_grid(obj.yGrid, padded_shape, fill=0)
        
        # 同步填充所有依赖该网格尺寸的辅助数组
        obj.Dx0   = _pad_grid(obj.Dx0, padded_shape, fill=0) if obj.Dx0 is not None else obj.Dx0
        obj.Dy0   = _pad_grid(obj.Dy0, padded_shape, fill=0) if obj.Dy0 is not None else obj.Dy0
        if isinstance(obj.SearchLimitX, np.ndarray):
            obj.SearchLimitX  = _pad_grid(obj.SearchLimitX, padded_shape, fill=0)
            obj.SearchLimitY  = _pad_grid(obj.SearchLimitY, padded_shape, fill=0)
        if isinstance(obj.ChipSizeMinX, np.ndarray):
            obj.ChipSizeMinX  = _pad_grid(obj.ChipSizeMinX, padded_shape, fill=0)
            obj.ChipSizeMaxX  = _pad_grid(obj.ChipSizeMaxX, padded_shape, fill=0)

    # ── 运行 autoRIFT ────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("运行 autoRIFT 特征追踪...")
    obj.runAutorift()
    log.info("autoRIFT 完成 (%.1f s)", time.time() - t0)

    # ── 扩张 noDataMask 消除边缘伪迹 ─────────────────────────────────────────
    kernel     = np.ones((3, 3), np.uint8)
    noDataMask = cv2.dilate(
        noDataMask.astype(np.uint8), kernel, iterations=1
    ).astype(bool)

    # ── 整理结果（截回原始格网尺寸）──────────────────────────────────────────
    grid_shape = _grid_shape_orig
    DX         = _pad_to(np.asarray(obj.Dx,         dtype=np.float32), grid_shape)
    DY         = _pad_to(np.asarray(obj.Dy,         dtype=np.float32), grid_shape)
    INTERPMASK = _pad_to(np.asarray(obj.InterpMask, dtype=np.float32), grid_shape)
    CHIPSIZEX  = _pad_to(np.asarray(obj.ChipSizeX,  dtype=np.float32), grid_shape)

    DX[noDataMask]         = np.nan
    DY[noDataMask]         = np.nan
    INTERPMASK[noDataMask] = 0
    CHIPSIZEX[noDataMask]  = 0
    if SRx0 is not None:
        out_of_range = (SRx0 == 0) | (SRx0 == NODATA_VAL)
        DX[out_of_range]         = np.nan
        DY[out_of_range]         = np.nan
        INTERPMASK[out_of_range] = 0

    # ── 偏移量 → 速度 ─────────────────────────────────────────────────────────
    log.info("偏移量转换为速度场（m/yr）...")
    VX, VY = _offsets_to_velocity(DX, DY, offset2vx_path, offset2vy_path, NODATA_VAL)

    # ── 速度阈值过滤（v3 新增）──────────────────────────────────────────────
    if velocity_max_m_yr > 0:
        exceed = (np.abs(VX) > velocity_max_m_yr) | (np.abs(VY) > velocity_max_m_yr)
        n_exceed = int(np.sum(exceed & np.isfinite(VX)))
        if n_exceed > 0:
            log.warning(
                "速度阈值过滤: %d 像素（|Vx|>%.0f 或 |Vy|>%.0f m/yr）置为 NoData",
                n_exceed, velocity_max_m_yr, velocity_max_m_yr,
            )
            VX[exceed] = np.nan
            VY[exceed] = np.nan

    V_MAG = np.full_like(VX, np.nan, dtype=np.float32)
    valid = np.isfinite(VX) & np.isfinite(VY)
    V_MAG[valid] = np.sqrt(VX[valid] ** 2 + VY[valid] ** 2)

    log.info(
        "速度统计（有效像素=%d / 总=%d）: |V| mean=%.1f  max=%.1f  [m/yr]",
        int(valid.sum()), valid.size,
        float(np.nanmean(V_MAG)) if valid.any() else 0.0,
        float(np.nanmax(V_MAG))  if valid.any() else 0.0,
    )

    # ── 误差估计（v3 新增）──────────────────────────────────────────────────
    log.info("计算 Vx / Vy 误差（稳定地表统计）...")
    ssm_window_tif = str(wd / "window_stable_surface_mask.tif")
    vx_error, vy_error, stable_count = _compute_errors(
        VX=VX,
        VY=VY,
        ssm_window_tif=ssm_window_tif,
        vx0_tif=vx0_tif,
        vy0_tif=vy0_tif,
        geo_transform=geo_transform,
        projection=projection,
    )
    v_error = (
        float(np.sqrt(vx_error ** 2 + vy_error ** 2))
        if (np.isfinite(vx_error) and np.isfinite(vy_error))
        else np.nan
    )
    log.info(
        "误差估计: Vx_err=%.2f  Vy_err=%.2f  V_err=%.2f  stable_count=%d  [m/yr]",
        vx_error if np.isfinite(vx_error) else -1,
        vy_error if np.isfinite(vy_error) else -1,
        v_error  if np.isfinite(v_error)  else -1,
        stable_count,
    )

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    results = {}

    vel_tif = str(Path(output_dir) / f"velocity_{pair_name}.tif")
    _save_velocity(vel_tif, VX, VY, V_MAG, geo_transform, projection)
    results["velocity_tif"] = vel_tif

    if save_offsets:
        off_tif = str(Path(output_dir) / f"offset_{pair_name}.tif")
        _save_offsets(off_tif, DX, DY, INTERPMASK, CHIPSIZEX, geo_transform, projection)
        results["offset_tif"] = off_tif

    if save_netcdf:
        nc_path = str(Path(output_dir) / f"velocity_{pair_name}.nc")
        _save_netcdf(
            path=nc_path,
            VX=VX, VY=VY, V_MAG=V_MAG,
            INTERPMASK=INTERPMASK, CHIPSIZEX=CHIPSIZEX,
            geo_transform=geo_transform,
            projection=projection,
            epsg=epsg,
            pair_name=pair_name,
            date_ref=date_ref,
            date_sec=date_sec,
            vx_error=vx_error,
            vy_error=vy_error,
            v_error=v_error,
            stable_count=stable_count,
            has_prior_velocity=(vx0_tif is not None and Path(vx0_tif).exists()),
        )
        results["netcdf"] = nc_path

    results.update({
        "vx_error":     vx_error,
        "vy_error":     vy_error,
        "v_error":      v_error,
        "stable_count": stable_count,
    })
    return results


# ── 误差估计 ─────────────────────────────────────────────────────────────────

def _compute_errors(
    VX: np.ndarray,
    VY: np.ndarray,
    ssm_window_tif: str,
    vx0_tif: Optional[str],
    vy0_tif: Optional[str],
    geo_transform: tuple,
    projection: str,
) -> Tuple[float, float, int]:
    """在稳定地表像素上估计 Vx / Vy 误差（标准差）。

    两种误差模式
    ──────────────────────────────────────────────────────────────
    有 vx0/vy0（SPS 参考速度）：
        Vx_Error = std( Vx_measured - Vx_reference )  over SSM pixels
        物理含义：测量值与已知真值的偏差，最能反映绝对精度。

    无 vx0/vy0（仅 SSM）：
        Vx_Error = std( Vx_measured )                 over SSM pixels
        物理含义：稳定地表真实速度约为 0，测量离散度即为噪声水平。
    ──────────────────────────────────────────────────────────────

    Returns
    -------
    (vx_error, vy_error, stable_count)  — 单位 m/yr；count 为有效稳定像素数
    """
    shape = VX.shape

    # 加载 window_stable_surface_mask（geogrid 坐标系，已与速度场对齐）
    if not Path(ssm_window_tif).exists():
        log.warning("window_stable_surface_mask.tif 不存在，跳过误差估计")
        return np.nan, np.nan, 0

    ssm_arr, _, _ = read_geotiff(ssm_window_tif)
    if ssm_arr.ndim == 3:
        ssm_arr = ssm_arr[0]
    # 对齐尺寸（以防 geogrid 输出与速度场有微小差异）
    r_min = min(ssm_arr.shape[0], shape[0])
    c_min = min(ssm_arr.shape[1], shape[1])
    ssm_mask = np.zeros(shape, dtype=bool)
    ssm_mask[:r_min, :c_min] = (ssm_arr[:r_min, :c_min] != 0)

    valid      = np.isfinite(VX) & np.isfinite(VY)
    stable_valid = ssm_mask & valid
    stable_count = int(stable_valid.sum())

    if stable_count < 10:
        log.warning("稳定地表有效像素过少（%d），误差估计不可靠", stable_count)
        return np.nan, np.nan, stable_count

    # ── 有参考速度：计算残差 ──────────────────────────────────────────────
    if (vx0_tif and Path(vx0_tif).exists()
            and vy0_tif and Path(vy0_tif).exists()):
        log.info("误差模式: 有 vx0/vy0 → 计算残差 std")
        vx0 = _resample_to_grid(vx0_tif, geo_transform, shape, projection, nodata=-32767.0)
        vy0 = _resample_to_grid(vy0_tif, geo_transform, shape, projection, nodata=-32767.0)

        # 参考速度本身也需有效
        ref_valid = np.isfinite(vx0) & np.isfinite(vy0)
        mask = stable_valid & ref_valid

        if mask.sum() < 10:
            log.warning("稳定地表上参考速度有效像素不足（%d），回退到无参考模式", mask.sum())
        else:
            vx_residual = (VX - vx0)[mask]
            vy_residual = (VY - vy0)[mask]
            return float(np.std(vx_residual)), float(np.std(vy_residual)), int(mask.sum())

    # ── 无参考速度：直接在稳定地表上计 std ───────────────────────────────
    log.info("误差模式: 无 vx0/vy0 → std(V) over SSM（稳定地表速度视为 0）")
    vx_stable = VX[stable_valid]
    vy_stable = VY[stable_valid]
    return float(np.std(vx_stable)), float(np.std(vy_stable)), stable_count


def _resample_to_grid(
    src_tif: str,
    geo_transform: tuple,
    shape: Tuple[int, int],
    projection: str,
    nodata: float = -32767.0,
) -> np.ndarray:
    """将外部栅格（如 SPS vx0）双线性重采样到速度场格网。

    目标格网由 geo_transform + shape（来自 window_location.tif）定义。
    """
    rows, cols = shape
    gt = geo_transform
    x_min = gt[0]
    x_max = gt[0] + gt[1] * cols
    y_max = gt[3]
    y_min = gt[3] + gt[5] * rows   # gt[5] < 0

    warp_opts = gdal.WarpOptions(
        format="MEM",
        width=cols, height=rows,
        outputBounds=(x_min, y_min, x_max, y_max),
        dstSRS=projection,
        srcNodata=nodata,
        dstNodata=np.nan,
        resampleAlg="bilinear",
    )
    ds  = gdal.Warp("", src_tif, options=warp_opts)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds  = None
    arr[arr == nodata] = np.nan
    return arr


# ── NetCDF 输出（CF-1.8）─────────────────────────────────────────────────────

def _save_netcdf(
    path: str,
    VX: np.ndarray,
    VY: np.ndarray,
    V_MAG: np.ndarray,
    INTERPMASK: np.ndarray,
    CHIPSIZEX: np.ndarray,
    geo_transform: tuple,
    projection: str,
    epsg: int,
    pair_name: str,
    date_ref: str,
    date_sec: str,
    vx_error: float,
    vy_error: float,
    v_error: float,
    stable_count: int,
    has_prior_velocity: bool,
) -> None:
    """将速度场写为 CF-1.8 NetCDF（xarray + netCDF4）。

    变量
    ────
    vx, vy, v         : 速度分量及量级（m/yr，float32，含 _FillValue）
    chip_size_max      : autoRIFT 实际使用的最大 chip 尺寸（m，float32）
    interp_mask        : 1=实测  0=插值/无效（uint8）

    全局属性（对齐 ITS_LIVE / hyp3 约定）
    ─────────────────────────────────────
    Conventions, title, date_created, pair_name, reference_date, secondary_date,
    dt_days, EPSG, crs_wkt,
    stable_count, error_method,
    vx_error, vy_error, v_error
    """
    try:
        import xarray as xr
    except ImportError:
        log.error("xarray 未安装，跳过 NetCDF 输出。请运行: pip install xarray netCDF4")
        return

    rows, cols = VX.shape
    gt = geo_transform

    # ── 坐标轴（像素中心）─────────────────────────────────────────────────
    x_coords = gt[0] + gt[1] * (np.arange(cols) + 0.5)
    y_coords = gt[3] + gt[5] * (np.arange(rows) + 0.5)

    # ── CRS WKT（从 projection 字符串或 EPSG 获取）────────────────────────
    srs = osr.SpatialReference()
    if projection:
        srs.ImportFromWkt(projection)
    else:
        srs.ImportFromEPSG(epsg)
    crs_wkt = srs.ExportToWkt()

    # ── 时差（天）────────────────────────────────────────────────────────
    try:
        d1 = datetime.strptime(date_ref, "%Y%m%d")
        d2 = datetime.strptime(date_sec, "%Y%m%d")
        dt_days = abs((d2 - d1).days)
    except (ValueError, TypeError):
        dt_days = -1

    # ── 误差估计方法标注 ──────────────────────────────────────────────────
    error_method = (
        "std(Vx - Vx_reference) over stable surface pixels"
        if has_prior_velocity
        else "std(Vx) over stable surface pixels (zero-velocity reference)"
    )

    def _fill(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        out[~np.isfinite(out)] = NODATA_VEL
        return out.astype(np.float32)

    # ── 构建 Dataset ──────────────────────────────────────────────────────
    ds = xr.Dataset(
        {
            "vx": xr.Variable(
                ["y", "x"], _fill(VX),
                attrs={
                    "units": "m/yr",
                    "long_name": "ice velocity component in x (easting) direction",
                    "_FillValue": NODATA_VEL,
                    "grid_mapping": "crs",
                },
            ),
            "vy": xr.Variable(
                ["y", "x"], _fill(VY),
                attrs={
                    "units": "m/yr",
                    "long_name": "ice velocity component in y (northing) direction",
                    "_FillValue": NODATA_VEL,
                    "grid_mapping": "crs",
                },
            ),
            "v": xr.Variable(
                ["y", "x"], _fill(V_MAG),
                attrs={
                    "units": "m/yr",
                    "long_name": "ice velocity magnitude",
                    "_FillValue": NODATA_VEL,
                    "grid_mapping": "crs",
                },
            ),
            "chip_size_max": xr.Variable(
                ["y", "x"], CHIPSIZEX.astype(np.float32),
                attrs={
                    "units": "m",
                    "long_name": "maximum chip size used in autoRIFT",
                    "_FillValue": 0.0,
                    "grid_mapping": "crs",
                },
            ),
            "interp_mask": xr.Variable(
                ["y", "x"], INTERPMASK.astype(np.uint8),
                attrs={
                    "long_name": "valid pixel flag (1=measured, 0=interpolated or no-data)",
                    "_FillValue": np.uint8(0),
                    "grid_mapping": "crs",
                },
            ),
            # CRS 变量（CF 惯例）
            "crs": xr.Variable(
                [], np.int32(epsg),
                attrs={
                    "grid_mapping_name": "universal_transverse_mercator",  # 占位，由 crs_wkt 覆盖
                    "crs_wkt": crs_wkt,
                    "epsg_code": f"EPSG:{epsg}",
                },
            ),
        },
        coords={
            "x": xr.Variable(
                "x", x_coords.astype(np.float64),
                attrs={"standard_name": "projection_x_coordinate", "units": "m"},
            ),
            "y": xr.Variable(
                "y", y_coords.astype(np.float64),
                attrs={"standard_name": "projection_y_coordinate", "units": "m"},
            ),
        },
        attrs={
            # ── CF 必填 ─────────────────────────────────────────────────
            "Conventions":       "CF-1.8",
            "title":             f"SAR ice velocity: {pair_name}",
            "date_created":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "institution":       "autoRIFT_S1_Local_Pipeline",
            "software":          "autoRIFT_S1_Local_Pipeline-v3 / autoRIFT",
            # ── 像对信息 ────────────────────────────────────────────────
            "pair_name":         pair_name,
            "reference_date":    date_ref,
            "secondary_date":    date_sec,
            "dt_days":           dt_days,
            # ── 投影 ────────────────────────────────────────────────────
            "EPSG":              epsg,
            # ── 误差 ────────────────────────────────────────────────────
            "stable_count":      stable_count,
            "error_method":      error_method,
            "vx_error":   round(float(vx_error), 3) if np.isfinite(vx_error) else float(NODATA_VEL),
            "vy_error":   round(float(vy_error), 3) if np.isfinite(vy_error) else float(NODATA_VEL),
            "v_error":    round(float(v_error),  3) if np.isfinite(v_error)  else float(NODATA_VEL),
        },
    )

    # ── 压缩编码 ───────────────────────────────────────────────────────────
    encoding = {
        var: {"zlib": True, "complevel": 4, "shuffle": True}
        for var in ["vx", "vy", "v", "chip_size_max", "interp_mask"]
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding, format="NETCDF4")
    log.info("NetCDF 已写入: %s", path)


# ── 辅助函数（保持原有实现）─────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    ds  = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开幅度图: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds  = None
    return arr


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
    """Load the stable-surface mask for later error statistics.

    Stable land is where a velocity result is needed to estimate bias and
    uncertainty, so this mask is not a matching no-data mask.
    """
    if not Path(ssm_path).exists():
        return np.zeros(shape, dtype=bool)
    arr, _, _ = read_geotiff(ssm_path)
    mask = (arr != 0).astype(bool)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != shape:
        mask = mask[:shape[0], :shape[1]]
    return mask


def _build_image_nodata_mask(
    I1: np.ndarray,
    I2: np.ndarray,
    xGrid: np.ndarray,
    yGrid: np.ndarray,
    init_mask: np.ndarray,
    nodata_val: int,
    build_zero_mask: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Add invalid window nodes caused by image bounds or zero amplitude.

    The old implementation iterated once per Geogrid node in Python.  At a
    30 m output grid this can mean more than 100 million iterations.  Process
    rows in vectorized blocks instead, keeping memory bounded for NISAR scenes.

    Returns the node no-data mask and, optionally, a full-image zero mask for
    preprocessing methods that need one.
    """
    mask = init_mask.copy()
    rows, cols = xGrid.shape
    h1, w1 = I1.shape

    zeroMask = ((I1 == 0) | (I2 == 0)) if build_zero_mask else None

    # A block keeps temporary arrays proportional to 512 times the output
    # width, instead of allocating full-grid advanced-index arrays.
    block_rows = 512
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        xs = xGrid[start:stop]
        ys = yGrid[start:stop]
        block_mask = mask[start:stop]

        valid = (
            (xs != nodata_val)
            & (ys != nodata_val)
            & (xs >= 1)
            & (ys >= 1)
            & (xs <= w1)
            & (ys <= h1)
        )
        block_mask[~valid] = True

        if valid.any():
            xr = xs[valid] - 1
            yr = ys[valid] - 1
            block_mask[valid] |= (I1[yr, xr] == 0) | (I2[yr, xr] == 0)

    return mask, zeroMask


def _compatible_chip_grid(
    chipsizex0: float,
    pixel_size_x: float,
    grid_spacing_m: float,
) -> Tuple[int, int, int, int]:
    """Return an autoRIFT-compatible base chip and grid spacing in pixels.

    Both values represent the same physical Geogrid configuration but become
    integers independently.  A 30 m DEM can therefore yield 52 px and 6 px,
    which violates the required ChipSize0X modulo GridSpacingX constraint.
    The returned base chip is snapped to the nearest multiple of the requested
    grid step and four pixels; the output Geogrid itself is untouched.
    """
    if not (np.isfinite(chipsizex0) and np.isfinite(pixel_size_x) and np.isfinite(grid_spacing_m)):
        raise ValueError("Geogrid ChipSize/像元/格网间距必须为有限数值")
    if chipsizex0 <= 0 or pixel_size_x <= 0 or grid_spacing_m <= 0:
        raise ValueError("Geogrid ChipSize/像元/格网间距必须为正数")

    requested_chip0 = int(np.ceil(chipsizex0 / pixel_size_x / 4.0) * 4)
    # Keep the upstream testautoRIFT relationship.  It intentionally derives
    # the sample step from the four-pixel-rounded base chip, rather than
    # independently rounding grid_spacing_m / pixel_size_x.
    requested_grid = max(
        1,
        int(requested_chip0 * grid_spacing_m / chipsizex0),
    )

    if requested_chip0 <= requested_grid:
        return requested_chip0, requested_grid, requested_chip0, requested_grid

    unit = abs(requested_grid * 4) // math.gcd(requested_grid, 4)
    lower = max(unit, (requested_chip0 // unit) * unit)
    upper = ((requested_chip0 + unit - 1) // unit) * unit
    chip0 = min((lower, upper), key=lambda value: (abs(value - requested_chip0), value))
    return chip0, requested_grid, requested_chip0, requested_grid


def _snap_chip_sizes(arr: np.ndarray, chip0: int, nodata: float) -> np.ndarray:
    """Snap chip sizes to the 1, 2, 4 and 8 nested-search scales."""
    out = np.zeros(arr.shape, dtype=np.int32)
    valid = (arr != nodata) & (arr > 0)
    if not valid.any():
        return out

    # Nearest boundaries between supported scales are 1.5, 3 and 6 times
    # ChipSize0X. Values above 6x map to the 8x layer.
    out[valid] = chip0
    np.copyto(out, chip0 * 2, where=valid & (arr >= chip0 * 1.5))
    np.copyto(out, chip0 * 4, where=valid & (arr >= chip0 * 3.0))
    np.copyto(out, chip0 * 8, where=valid & (arr >= chip0 * 6.0))
    return out


def _largest_chip_scale(chip_sizes: np.ndarray, chip0: int) -> int:
    """Return the largest enabled power-of-two nested-search scale."""
    if chip0 <= 0:
        return 1
    valid = chip_sizes[chip_sizes > 0]
    if valid.size == 0:
        return 1
    return max(1, int(np.max(valid) // chip0))


def _shape_multiple(shape: tuple, multiple: int) -> tuple:
    multiple = max(1, int(multiple))
    return tuple(int(np.ceil(length / multiple) * multiple) for length in shape)


def _pad_grid(arr: np.ndarray, shape: tuple, fill: int = 0) -> np.ndarray:
    if arr.shape == shape:
        return arr
    out = np.full(shape, fill, dtype=arr.dtype)
    out[:arr.shape[0], :arr.shape[1]] = arr
    return out


def _apply_filter(obj, filter_type: str, filter_width: int) -> None:
    # ft = (filter_type or "WAL").upper()   # None → 默认 WAL
    ft = filter_type.upper()   # 调用方已保证 filter_type 非空
    if ft == "WAL":
        obj.WallisFilterWidth = filter_width
        obj.preprocess_filt_wal()
    elif ft == "HPS":
        obj.preprocess_filt_hps()
    elif ft == "SOB":
        obj.preprocess_filt_sob()
    elif ft == "LAP":
        obj.preprocess_filt_lap()
    else:
        log.warning("未知滤波类型 '%s'，使用 HPS 高通滤波", filter_type)
        obj.preprocess_filt_hps()


def _offsets_to_velocity(
    DX: np.ndarray,
    DY: np.ndarray,
    vx_path: str,
    vy_path: str,
    nodata_val: float,
) -> tuple:
    """像素偏移 (DX, DY) → 地理速度 (Vx, Vy) m/yr。"""
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

    DX64 = DX.astype(np.float64)
    DY64 = DY.astype(np.float64)

    VX = (vx1 * DX64 + vx2 * DY64).astype(np.float32)
    VY = (vy1 * DX64 + vy2 * DY64).astype(np.float32)
    VX[~np.isfinite(DX64)] = np.nan
    VY[~np.isfinite(DY64)] = np.nan
    return VX, VY


def _save_velocity(path, VX, VY, V, geo_transform, projection):
    def _fill(a):
        x = a.copy(); x[~np.isfinite(x)] = NODATA_VEL; return x.astype(np.float32)
    write_geotiff(
        path, [_fill(VX), _fill(VY), _fill(V)],
        geo_transform, projection,
        nodata=NODATA_VEL,
        band_names=["Vx_m_yr", "Vy_m_yr", "V_m_yr"],
    )
    log.info("速度场已写入: %s", path)


def _save_offsets(path, DX, DY, INTERPMASK, CHIPSIZEX, geo_transform, projection):
    def _fill(a, nv):
        x = a.copy(); x[~np.isfinite(x)] = nv; return x.astype(np.float32)
    write_geotiff(
        path,
        [_fill(DX, NODATA_OFFSET), _fill(DY, NODATA_OFFSET),
         _fill(INTERPMASK, 0),      _fill(CHIPSIZEX, 0)],
        geo_transform, projection,
        nodata=NODATA_OFFSET,
        band_names=["DX_px", "DY_px", "InterpMask", "ChipSizeX"],
    )
    log.info("偏移量已写入: %s", path)


def _pad_to(arr: np.ndarray, shape: tuple) -> np.ndarray:
    if arr.shape == shape:
        return arr
    out = np.full(shape, np.nan, dtype=np.float32)
    r = min(arr.shape[0], shape[0])
    c = min(arr.shape[1], shape[1])
    out[:r, :c] = arr[:r, :c]
    return out
