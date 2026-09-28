"""
optical_processor.py — 光学影像数据接入、预处理与配准元数据提取

对应 S1 pipeline 中的 s1_processor.py，但光学链路不需要：
  - COMPASS/CSLC 处理
  - 轨道文件
  - Burst 拼接

光学链路替代步骤：
  1. 本地波段文件定位（B08/B8/B2）
  2. 平台探测与日期解析
  3. 平台专属预处理滤波（L4/5 FFT, L7 Wallis-fill, S2/L8/L9 HPS）
  4. GeogridOptical.coregister() 空间配准 + 元数据提取

【平台对应关系】
  S2:    B08.jp2   (NIR 10m)         → HPS 预处理（内部 autoRIFT 完成）
  L8/L9: B8.TIF    (全色 15m)        → HPS 预处理
  L7:    B8.TIF    (全色 15m, SLC-off)→ Wallis-fill 预处理（外部，Geogrid 前完成）
  L4/L5: B2.TIF    (绿波段 30m)      → FFT 预处理（外部，Geogrid 前完成）

【坐标体系差异 vs. SAR】
  SAR:    radar 坐标 (range/azimuth)，需 Geogrid 转 map 坐标
  Optical: 直接 map 坐标 (GeoTIFF 带 GeoTransform)，Geogrid 做像素→速度映射
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 平台标识与命名规则
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel-2 ESA 格式：S2A_MSIL1C_YYYYMMDDTHHMMSS_...
_RE_S2_ESA  = re.compile(r'^S2[AB]_MSIL\d C_(\d{8})T\d{6}', re.IGNORECASE)
# Sentinel-2 COG 格式：S2A_..._YYYYMMDD_...
_RE_S2_COG  = re.compile(r'^S2[AB]_\w+_(\d{8})_', re.IGNORECASE)
# Landsat Collection 2：LC09/LC08/LE07/LT05/LT04_PPPRR_YYYYMMDD_...
_RE_LANDSAT = re.compile(r'^L[CETO][0-9][45789]_\d{6}_(\d{8})_', re.IGNORECASE)

# 平台正则（用于从文件名识别传感器）
_PLATFORM_PATTERNS: Dict[str, re.Pattern] = {
    "L9": re.compile(r"L[CO]09_"),
    "L8": re.compile(r"L[CO]08_"),
    "L7": re.compile(r"L[EO]07_"),
    "L5": re.compile(r"LT05_"),
    "L4": re.compile(r"LT04_"),
    "S2": re.compile(r"S2[AB]_"),
}

# 各平台对应的波段文件名后缀（用于 find_local_band）
_BAND_SUFFIX: Dict[str, Tuple[str, ...]] = {
    "S2": ("_B08.jp2", "_B08.tif", "_B8.jp2"),
    "L9": ("_B8.TIF", "_B8.tif"),
    "L8": ("_B8.TIF", "_B8.tif"),
    "L7": ("_B8.TIF", "_B8.tif"),
    "L5": ("_B2.TIF", "_B2.tif"),
    "L4": ("_B2.TIF", "_B2.tif"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. 平台 & 日期工具
# ─────────────────────────────────────────────────────────────────────────────

def detect_platform(path: str) -> str:
    """从文件路径/文件名识别光学平台。

    Returns
    -------
    str  "S2" | "L9" | "L8" | "L7" | "L5" | "L4"
    """
    name = Path(path).name
    for platform, pat in _PLATFORM_PATTERNS.items():
        if pat.search(name):
            return platform
    raise ValueError(
        f"无法识别平台：{name}\n"
        "文件名应符合 Sentinel-2 或 Landsat Collection 2 命名规范。"
    )


def get_scene_date(path: str) -> str:
    """从文件路径提取采集日期 YYYYMMDD。"""
    name = Path(path).name
    # S2 ESA
    m = re.search(r"_(\d{8})T\d{6}_", name)
    if m:
        return m.group(1)
    # S2 COG / Landsat
    m = re.search(r"_(\d{8})_", name)
    if m:
        return m.group(1)
    raise ValueError(f"无法从文件名提取日期：{name}")


def get_scene_name(path: str) -> str:
    """从文件路径提取场景名（不含波段后缀）。"""
    name = Path(path).name
    platform = detect_platform(path)
    for suffix in _BAND_SUFFIX.get(platform, ()):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    # 回退：去掉扩展名
    return Path(path).stem


def find_local_band(scene_dir: str, scene_name: str, platform: str) -> str:
    """在本地目录中查找对应平台的波段文件。

    Parameters
    ----------
    scene_dir  : 场景文件夹根目录
    scene_name : 场景名（不含波段后缀）
    platform   : "S2" | "L8" | ...

    Returns
    -------
    str  波段文件的绝对路径
    """
    scene_path = Path(scene_dir)
    for suffix in _BAND_SUFFIX.get(platform, ()):
        candidates = list(scene_path.rglob(f"{scene_name}{suffix}"))
        if candidates:
            return str(candidates[0].resolve())
    raise FileNotFoundError(
        f"在 {scene_dir} 中未找到场景 {scene_name} 的波段文件。\n"
        f"期望后缀: {_BAND_SUFFIX.get(platform, ())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. 预处理滤波（仅 L4/L5/L7 需要外部预处理；S2/L8/L9 由 autoRIFT 内部完成）
# ─────────────────────────────────────────────────────────────────────────────

def _apply_fft_filter(path: str, work_dir: str) -> str:
    """FFT + Wallis 滤波（L4/L5）。
    
    保存到 {work_dir}/filtered/{basename}，文件名保持不变（确保 Geogrid 能识别平台）。
    """
    try:
        from autoRIFT.autoRIFT import _fft_filter, _wallis_filter
    except ImportError:
        raise ImportError("autoRIFT 未安装，无法运行 FFT 滤波。")

    out_dir = Path(work_dir) / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / Path(path).name)

    ds  = gdal.Open(path, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue() or 0
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds   = None

    valid = arr != nodata
    arr[~valid] = 0

    wallis = _wallis_filter(arr, filter_width=5)
    wallis[~valid] = 0

    filtered = _fft_filter(wallis, valid, power_threshold=500)
    filtered[~valid] = 0

    _write_filtered(out_path, filtered.astype(np.float32), gt, proj)
    log.info("FFT 滤波完成: %s", out_path)
    return out_path


def _apply_wallis_fill_filter(path: str, work_dir: str) -> Tuple[str, str]:
    """Wallis + 零值填充滤波（L7 SLC-off）。

    Returns
    -------
    (filtered_path, zero_mask_path)
    """
    try:
        from autoRIFT.autoRIFT import _wallis_filter_fill
    except ImportError:
        raise ImportError("autoRIFT 未安装，无法运行 Wallis 滤波。")

    out_dir = Path(work_dir) / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    out_path  = str(out_dir / Path(path).name)
    mask_path = str(out_dir / f"{stem}_zeroMask.tif")

    ds  = gdal.Open(path, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue() or 0
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds   = None

    arr[arr == nodata] = 0

    filtered, zero_mask = _wallis_filter_fill(arr, filter_width=5, std_cutoff=0.25)
    filtered[zero_mask] = 0

    _write_filtered(out_path, filtered.astype(np.float32), gt, proj)
    _write_mask(mask_path, zero_mask.astype(np.uint8), gt, proj)
    log.info("Wallis-fill 滤波完成: %s (零值掩膜: %s)", out_path, mask_path)
    return out_path, mask_path


def _write_filtered(path: str, arr: np.ndarray, gt: tuple, proj: str) -> None:
    rows, cols = arr.shape
    drv = gdal.GetDriverByName("GTiff")
    ds  = drv.Create(path, cols, rows, 1, gdal.GDT_Float32,
                     options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def _write_mask(path: str, arr: np.ndarray, gt: tuple, proj: str) -> None:
    rows, cols = arr.shape
    drv = gdal.GetDriverByName("GTiff")
    ds  = drv.Create(path, cols, rows, 1, gdal.GDT_Byte,
                     options=["COMPRESS=LZW"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).SetNoDataValue(255)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def apply_preprocessing(
    ref_path: str,
    sec_path: str,
    platform: str,
    work_dir: str,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """平台专属预处理滤波（Geogrid 调用之前执行）。

    Returns
    -------
    ref_out, sec_out : 经滤波的影像路径（S2/L8/L9 直接返回原路径）
    ref_zero_mask, sec_zero_mask : 零值掩膜路径（仅 L7 非 None）

    【说明】
    S2 / L8 / L9 — 使用 HPS 高通滤波，由 autoRIFT 内部完成；此处直接透传。
    L7            — Wallis-fill 滤波（SLC-off 条纹处理），外部必须完成。
    L4 / L5       — FFT + Wallis 滤波，外部必须完成。
    """
    ref_zero = None
    sec_zero = None

    if platform in ("S2", "L8", "L9"):
        log.info("平台 %s：使用 HPS 高通滤波（由 autoRIFT 内部完成，此处透传）", platform)
        return ref_path, sec_path, ref_zero, sec_zero

    elif platform == "L7":
        log.info("平台 L7 (SLC-off)：应用 Wallis-fill 外部预处理...")
        ref_out, ref_zero = _apply_wallis_fill_filter(ref_path, work_dir)
        sec_out, sec_zero = _apply_wallis_fill_filter(sec_path, work_dir)
        return ref_out, sec_out, ref_zero, sec_zero

    elif platform in ("L4", "L5"):
        log.info("平台 %s：应用 FFT + Wallis 外部预处理...", platform)
        ref_out = _apply_fft_filter(ref_path, work_dir)
        sec_out = _apply_fft_filter(sec_path, work_dir)
        return ref_out, sec_out, ref_zero, sec_zero

    else:
        raise ValueError(f"未支持的平台: {platform}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 配准元数据提取（wraps GeogridOptical.coregister()）
# ─────────────────────────────────────────────────────────────────────────────

def load_optical_metadata_pair(
    ref_path: str,
    sec_path: str,
    ref_name: Optional[str] = None,
    sec_name: Optional[str] = None,
) -> Tuple[object, object]:
    """提取光学影像对的配准元数据（对应 S1 pipeline 中的 load_slc_metadata_pair）。

    使用 GeogridOptical.coregister() 找出两景的公共覆盖区，提取：
      info.startingX / startingY   — 左上角地图坐标（X/Y）
      info.XSize / YSize            — 像素分辨率（米）
      info.numberOfLines / Samples  — 配准后公共区域尺寸
      info.filename                 — 参考景路径（loadProductOptical 使用）
      info.time                     — YYYYMMDD（计算 repeatTime 用）

    Parameters
    ----------
    ref_path, sec_path : 经预处理的波段文件路径
    ref_name, sec_name : 可选场景名（覆盖自动提取的 time 字段）
    """
    try:
        from geogrid import GeogridOptical
    except ImportError:
        raise ImportError("geogrid 未安装，请确保 autoRIFT 环境已激活。")

    log.info("运行 GeogridOptical.coregister()...")
    obj = GeogridOptical()
    x1a, y1a, xsize1, ysize1, x2a, y2a, xsize2, ysize2, trans = obj.coregister(
        ref_path, sec_path
    )
    log.info(
        "配准公共区域: ref=(%d,%d)+%dx%d  sec=(%d,%d)+%dx%d",
        x1a, y1a, xsize1, ysize1, x2a, y2a, xsize2, ysize2,
    )

    class _MetaOptical:
        pass

    # ── 参考景元数据 ──────────────────────────────────────────────────────────
    meta_r = _MetaOptical()
    meta_r.startingX      = trans[0]
    meta_r.startingY      = trans[3]
    meta_r.XSize          = trans[1]   # 正值，地图坐标 x 分辨率
    meta_r.YSize          = trans[5]   # 负值，地图坐标 y 分辨率（北半球 < 0）
    meta_r.numberOfLines  = ysize1
    meta_r.numberOfSamples = xsize1
    meta_r.filename       = ref_path

    # 时间字段（YYYYMMDD，用于计算 repeatTime）
    ref_time = _extract_time(ref_path, ref_name)
    meta_r.time = ref_time
    log.info("参考景时间: %s", ref_time)

    # 记录公共区读取窗口（供 loadProductOptical 直接使用）
    meta_r._x1a   = x1a
    meta_r._y1a   = y1a
    meta_r._xsize = xsize1
    meta_r._ysize = ysize1

    # ── 次景元数据 ──────────────────────────────────────────────────────────
    meta_s = _MetaOptical()
    meta_s.startingX      = trans[0]
    meta_s.startingY      = trans[3]
    meta_s.XSize          = trans[1]
    meta_s.YSize          = trans[5]
    meta_s.numberOfLines  = ysize2
    meta_s.numberOfSamples = xsize2
    meta_s.filename       = sec_path

    sec_time = _extract_time(sec_path, sec_name)
    meta_s.time = sec_time
    log.info("次景时间:   %s", sec_time)

    meta_s._x1a   = x2a
    meta_s._y1a   = y2a
    meta_s._xsize = xsize2
    meta_s._ysize = ysize2

    return meta_r, meta_s


def _extract_time(path: str, override_name: Optional[str] = None) -> str:
    """从路径或覆盖名中提取日期字符串 YYYYMMDD（testGeogrid.py 使用此格式）。"""
    name = override_name or Path(path).name
    # S2 ESA / Landsat: 找 _YYYYMMDD_ 或 _YYYYMMDDTHHMMSS_
    m = re.search(r"_(\d{8})T?\d*_", name)
    if m:
        return m.group(1)
    m = re.search(r"_(\d{8})_", name)
    if m:
        return m.group(1)
    # 最后回退：直接返回 GDAL GetDescription 并让 testGeogrid 处理
    return get_scene_date(path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 影像加载（配准后读取公共区域）
# ─────────────────────────────────────────────────────────────────────────────

def load_optical_image_pair(
    ref_path: str,
    sec_path: str,
    meta_r,
    meta_s,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载配准后的光学影像对（对应 hyp3 的 loadProductOptical）。

    使用 meta_r/meta_s 中存储的公共区窗口偏移量读取，
    确保两景严格空间对齐后传入 autoRIFT。
    """
    log.info("加载参考景影像 (窗口: x=%d y=%d w=%d h=%d)...",
             meta_r._x1a, meta_r._y1a, meta_r._xsize, meta_r._ysize)
    ds1 = gdal.Open(ref_path, gdal.GA_ReadOnly)
    I1  = ds1.ReadAsArray(
        xoff=meta_r._x1a, yoff=meta_r._y1a,
        xsize=meta_r._xsize, ysize=meta_r._ysize,
    ).astype(np.float32)
    ds1 = None

    log.info("加载次景影像 (窗口: x=%d y=%d w=%d h=%d)...",
             meta_s._x1a, meta_s._y1a, meta_s._xsize, meta_s._ysize)
    ds2 = gdal.Open(sec_path, gdal.GA_ReadOnly)
    I2  = ds2.ReadAsArray(
        xoff=meta_s._x1a, yoff=meta_s._y1a,
        xsize=meta_s._xsize, ysize=meta_s._ysize,
    ).astype(np.float32)
    ds2 = None

    log.info("参考景尺寸: %s | 次景尺寸: %s", I1.shape, I2.shape)
    return I1, I2


# ─────────────────────────────────────────────────────────────────────────────
# 5. 多景 VRT 拼接（可选，对应 hyp3 的 mosaic 逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def build_vrt_mosaic(paths: list, output_path: str) -> str:
    """将多个同平台同日期影像拼接为 VRT 虚拟影像。

    用于覆盖范围跨多景的情况（如南极宽轨道 S2 条带）。
    """
    if len(paths) == 1:
        return paths[0]
    log.info("构建 VRT 拼接图 (%d 景) → %s", len(paths), output_path)
    gdal.BuildVRT(output_path, paths)
    return output_path
