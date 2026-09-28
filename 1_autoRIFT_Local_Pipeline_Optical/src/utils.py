"""
utils.py — 公共工具函数
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
log = logging.getLogger(__name__)


# ── 日志 ──────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


# ── GeoTIFF 读写 ───────────────────────────────────────────────────────────────

def read_geotiff(path: str) -> Tuple[np.ndarray, tuple, str]:
    """读取 GeoTIFF，返回 (array, geo_transform, projection)。
    多波段时 array.shape = (bands, rows, cols)，单波段 = (rows, cols)。
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开: {path}")
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    if ds.RasterCount == 1:
        arr = ds.GetRasterBand(1).ReadAsArray()
    else:
        arr = ds.ReadAsArray()  # shape (bands, rows, cols)
    ds = None
    return arr, geo_transform, projection


def write_geotiff(
    path: str,
    arrays: List[np.ndarray],
    geo_transform: tuple,
    projection: str,
    nodata: Optional[float] = None,
    band_names: Optional[List[str]] = None,
) -> None:
    """将一组 numpy 数组写入多波段 GeoTIFF。

    Parameters
    ----------
    path : str
        输出路径
    arrays : list of 2-D ndarray
        每个元素对应一个波段
    geo_transform : tuple
        GDAL GeoTransform (6-tuple)
    projection : str
        WKT 投影字符串
    nodata : float, optional
        无数据值
    band_names : list of str, optional
        波段名称
    """
    if not arrays:
        raise ValueError("arrays 列表为空")

    rows, cols = arrays[0].shape
    n_bands = len(arrays)

    # 确定 GDAL 数据类型
    dtype_map = {
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
        np.int16:   gdal.GDT_Int16,
        np.int32:   gdal.GDT_Int32,
        np.uint8:   gdal.GDT_Byte,
        np.uint16:  gdal.GDT_UInt16,
    }
    gdal_dtype = dtype_map.get(arrays[0].dtype.type, gdal.GDT_Float32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, cols, rows, n_bands, gdal_dtype,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(projection)

    for i, arr in enumerate(arrays, start=1):
        band = ds.GetRasterBand(i)
        band.WriteArray(arr.astype(arrays[0].dtype))
        if nodata is not None:
            band.SetNoDataValue(nodata)
        if band_names and i <= len(band_names):
            band.SetDescription(band_names[i - 1])
        band.FlushCache()

    ds.FlushCache()
    ds = None
    log.debug("已写入 %s (%d 波段)", path, n_bands)


def get_epsg_from_file(path: str) -> int:
    """从 GeoTIFF / GDAL 可读文件中提取 EPSG 代码。"""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开: {path}")
    srs = osr.SpatialReference(ds.GetProjection())
    ds = None
    srs.AutoIdentifyEPSG()
    code = srs.GetAuthorityCode(None)
    if code is None:
        raise ValueError(f"无法从 {path} 自动识别 EPSG 代码，请在 config.yaml 中手动指定 epsg。")
    return int(code)


def get_raster_info(path: str) -> dict:
    """返回栅格基本信息字典。"""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    info = {
        "cols":       ds.RasterXSize,
        "rows":       ds.RasterYSize,
        "bands":      ds.RasterCount,
        "geo_transform": gt,
        "projection": ds.GetProjection(),
        "x_res":      gt[1],
        "y_res":      abs(gt[5]),
        "x_min":      gt[0],
        "y_max":      gt[3],
        "x_max":      gt[0] + gt[1] * ds.RasterXSize,
        "y_min":      gt[3] + gt[5] * ds.RasterYSize,
    }
    ds = None
    return info


# ── 目录管理 ──────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_path(path: Optional[str]) -> Optional[str]:
    """将路径转为绝对路径字符串，None 时返回空字符串。"""
    if path is None:
        return ""
    return str(Path(path).resolve())
