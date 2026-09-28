"""
nisar_processor.py — NISAR RSLC/GSLC → 幅度图 + Geogrid 元数据

参考：hyp3_autorift/nisar_isce3.py（官方 NISAR 处理模块，已深度研读）
适配：
  ① 完全本地化——去除 ASF/S3 下载，用户提供本地 .h5 文件路径
  ② 接口对齐 autoRIFT_SAR_Local_Pipeline：
       - 与 s1_processor.py 接口风格一致
       - 与 geogrid_builder.run_geogrid_sar / autorift_runner.run_autorift 衔接
  ③ 目录结构对齐：
       results/{pair_name}/
         ├── scratch/            ← ISCE3 协同配准临时输出（RSLC）
         ├── nisar_ref_cropped.tif   ← 裁剪中间文件（GSLC）
         ├── nisar_sec_cropped.tif
         ├── reference.tif       ← 最终幅度图（所有类型）
         └── secondary.tif

支持产品类型：
  RSLC（Range-azimuth）
    1. ISCE3 协同配准次景：rdr2geo → geo2rdr → resample_slc
    2. 提取幅度（CFloat32 → float32 幅度），HPS 滤波，写 reference.tif / secondary.tif
    3. loadMetadataRslc → (meta_r, meta_s)；optical_flag = 0

  GSLC（地理编码 SLC）
    1. 裁剪参考/次景至空间重叠区
    2. 提取幅度，HPS 滤波，保留地理坐标，写 reference.tif / secondary.tif
    3. GSLCMetadata → (meta_r, meta_s)；optical_flag = 1

依赖（仅 NISAR 处理时需要）：
  nisar           — ISCE3 Python 接口（conda: isce3）
  cv2             — OpenCV（幅度图 HPS 滤波）
  osgeo.gdal      — 标准 GDAL 读写
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
log = logging.getLogger(__name__)

# ── 全局常量 ──────────────────────────────────────────────────────────────────
WALLIS_FILTER_WIDTH = 21   # HPS 高通核宽度（官方默认值）
BLOCK_SIZE          = 10000  # 分块读取行数（防内存溢出）


# ══════════════════════════════════════════════════════════════════════════════
# 1. 产品类型 & 极化检测
# ══════════════════════════════════════════════════════════════════════════════

def detect_nisar_type(h5_path: str) -> str:
    """从 HDF5 文件名或内部结构识别 NISAR 产品类型。

    Returns
    -------
    'RSLC' 或 'GSLC'
    """
    name = Path(h5_path).name.upper()
    for ptype in ('RSLC', 'GSLC'):
        if ptype in name:
            log.info("NISAR 产品类型（文件名）: %s", ptype)
            return ptype

    # 文件名无法判断时，尝试探测 HDF5 内部路径
    for ptype in ('RSLC', 'GSLC'):
        test_key = {
            'RSLC': f'HDF5:{h5_path}://science/LSAR/RSLC/swaths',
            'GSLC': f'HDF5:{h5_path}://science/LSAR/GSLC/grids',
        }[ptype]
        ds = gdal.Open(test_key)
        if ds is not None:
            ds = None
            log.info("NISAR 产品类型（HDF5 内部）: %s", ptype)
            return ptype

    raise ValueError(
        f"无法识别 NISAR 产品类型（RSLC / GSLC）: {h5_path}\n"
        "请确认文件名包含 'RSLC' 或 'GSLC'，或检查 HDF5 内部结构。"
    )


def detect_polarization(h5_path: str, frequency: str = 'A') -> str:
    """检测可用极化，优先选 HH，其次 VV，否则取第一个。"""
    try:
        from nisar.products.readers import product as nisar_product
        slc  = nisar_product.open_product(h5_path)
        pols = slc.polarizations.get(frequency, [])
        log.info("NISAR 可用极化 (freq=%s): %s", frequency, pols)
        for prefer in ('HH', 'VV'):
            if prefer in pols:
                return prefer
        if pols:
            return pols[0]
    except Exception as e:
        log.warning("nisar.products.readers 不可用 (%s)，默认选 HH", e)
    return 'HH'


# ══════════════════════════════════════════════════════════════════════════════
# 2. GSLC 专用：空间重叠裁剪
# ══════════════════════════════════════════════════════════════════════════════

def _raster_bounds(ds) -> Tuple[float, float, float, float, tuple]:
    """返回 (xmin, ymin, xmax, ymax, geotransform)。"""
    gt   = ds.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * ds.RasterXSize
    ymin = ymax + gt[5] * ds.RasterYSize
    return xmin, ymin, xmax, ymax, gt


def _srcwin(gt: tuple, xmin: float, ymin: float,
            xmax: float, ymax: float) -> Tuple[int, int, int, int]:
    """根据地理坐标范围计算 GDAL srcWin (xoff, yoff, xsize, ysize)。"""
    xoff  = int(round((xmin - gt[0]) / gt[1]))
    yoff  = int(round((ymax - gt[3]) / gt[5]))
    xsize = int(round((xmax - xmin) / gt[1]))
    ysize = int(round((ymin - ymax) / gt[5]))
    return xoff, yoff, max(xsize, 1), max(ysize, 1)


def crop_gslcs_overlap(
    ref_h5: str,
    sec_h5: str,
    work_dir: str,
    frequency:    str = 'A',
    polarization: str = 'HH',
) -> Tuple[str, str]:
    """将 GSLC 参考/次景裁剪至空间重叠区，输出临时 GeoTIFF。

    Returns
    -------
    (ref_cropped_path, sec_cropped_path) — GDAL 可读的 GeoTIFF 路径
    """
    ref_key = f'HDF5:{ref_h5}://science/LSAR/GSLC/grids/frequency{frequency}/{polarization}'
    sec_key = f'HDF5:{sec_h5}://science/LSAR/GSLC/grids/frequency{frequency}/{polarization}'

    ds1 = gdal.Open(ref_key)
    ds2 = gdal.Open(sec_key)
    if ds1 is None or ds2 is None:
        raise FileNotFoundError(
            f"无法打开 GSLC HDF5 波段:\n  参考: {ref_key}\n  次景: {sec_key}"
        )

    x1min, y1min, x1max, y1max, gt1 = _raster_bounds(ds1)
    x2min, y2min, x2max, y2max, gt2 = _raster_bounds(ds2)

    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "GSLC 参考景与次景无空间重叠，无法裁剪。\n"
            f"  参考: [{x1min:.1f},{y1min:.1f},{x1max:.1f},{y1max:.1f}]\n"
            f"  次景: [{x2min:.1f},{y2min:.1f},{x2max:.1f},{y2max:.1f}]"
        )

    out1 = str(Path(work_dir) / 'nisar_ref_cropped.tif')
    out2 = str(Path(work_dir) / 'nisar_sec_cropped.tif')

    gdal.Translate(out1, ds1, srcWin=_srcwin(gt1, xmin, ymin, xmax, ymax))
    gdal.Translate(out2, ds2, srcWin=_srcwin(gt2, xmin, ymin, xmax, ymax))
    ds1 = ds2 = None

    log.info(
        "GSLC 裁剪完成: 重叠区 [%.1f, %.1f, %.1f, %.1f]",
        xmin, ymin, xmax, ymax,
    )
    return out1, out2


# ══════════════════════════════════════════════════════════════════════════════
# 3. 幅度提取（CFloat32 → uint8，HPS 滤波）
#    对齐官方 convert_slc_to_uint8_amplitude
# ══════════════════════════════════════════════════════════════════════════════

def _hps_filter(img: np.ndarray, width: int) -> np.ndarray:
    """高通 (HPS / Wallis) 滤波核，与官方实现完全一致。"""
    k = -np.ones((width, width), dtype=np.float32)
    k[width // 2, width // 2] = k.size - 1
    k /= k.size
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_CONSTANT)


def extract_amplitude(
    gdal_path:           str,
    out_tif:             str,
    is_geocoded:         bool = False,
    wallis_filter_width: int  = WALLIS_FILTER_WIDTH,
) -> None:
    """从 GDAL 可读路径提取幅度并写出 GeoTIFF（uint8）。

    流程
    ----
    1. 分块读取 CFloat32 → |z|（幅度）
    2. HPS 高通滤波（官方 Wallis 核）
    3. 归一化到 [0, 255]（mean ± 3σ 拉伸）
    4. 写出 GeoTIFF（uint8；GSLC 携带地理坐标）

    Parameters
    ----------
    gdal_path    : GDAL 可读路径（如 'HDF5:file.h5://...path...' 或普通 .slc）
    out_tif      : 输出 GeoTIFF 路径
    is_geocoded  : True → GSLC，保留 GT / 投影；False → RSLC，无地理坐标
    """
    t0 = time.time()
    log.info("提取幅度: %s → %s", gdal_path, Path(out_tif).name)

    ds = gdal.Open(gdal_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开 NISAR 数据路径: {gdal_path}")

    gt   = ds.GetGeoTransform(can_return_null=True)
    proj = ds.GetProjectionRef()
    band = ds.GetRasterBand(1)
    nrows, ncols = band.YSize, band.XSize

    # ── 分块读取 CFloat32 → float32 幅度 ─────────────────────────────────
    img  = np.zeros((nrows, ncols), dtype=np.float32)
    blk  = BLOCK_SIZE
    n_blk = (nrows + blk - 1) // blk
    for i, row in enumerate(range(0, nrows, blk), 1):
        rows_this = min(blk, nrows - row)
        raw = band.ReadRaster(
            xoff=0, yoff=row, xsize=ncols, ysize=rows_this,
            buf_xsize=ncols, buf_ysize=rows_this,
            buf_type=gdal.GDT_CFloat32,
        )
        img[row:row + rows_this] = (
            np.abs(np.frombuffer(raw, np.complex64))
            .reshape(rows_this, ncols)
            .astype(np.float32)
        )
        log.debug("  块 %d/%d 读取完成", i, n_blk)
    ds = None

    # GSLC 特殊处理：NaN/Inf → 0
    if is_geocoded:
        img[~np.isfinite(img)] = 0.0

    valid = img != 0.0

    # ── HPS 高通滤波 ──────────────────────────────────────────────────────
    img = _hps_filter(img, wallis_filter_width)

    # ── 归一化到 [0, 255] ─────────────────────────────────────────────────
    vvals = img[valid]
    if vvals.size > 1:
        S = np.std(vvals) * np.sqrt(vvals.size / max(vvals.size - 1.0, 1.0))
        M = np.mean(vvals)
        img -= M - 3 * S
        img /= max(6 * S, 1e-8)
        img *= 256.0

    np.clip(img, 0, 255, out=img)
    np.rint(img, out=img)

    if is_geocoded:
        img[~np.isfinite(img)] = 0.0
    img[~valid] = 0.0

    # ── 写出 GeoTIFF (uint8) ──────────────────────────────────────────────
    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    drv    = gdal.GetDriverByName('GTiff')
    out_ds = drv.Create(
        out_tif, ncols, nrows, 1, gdal.GDT_Byte,
        options=['COMPRESS=LZW'],
    )
    if is_geocoded and gt:
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(img.astype(np.uint8))
    out_ds.FlushCache()
    out_ds = None

    log.info("幅度提取完成 (%.1f s): %s", time.time() - t0, out_tif)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RSLC 专用：ISCE3 协同配准
# ══════════════════════════════════════════════════════════════════════════════

def run_isce3_coreg(
    reference:    str,
    secondary:    str,
    dem_file:     str,
    work_dir:     str,
    frequency:    str = 'A',
    polarization: str = 'HH',
    resample_type: str = 'coarse',
) -> None:
    """使用 ISCE3 对 NISAR RSLC 次景进行协同配准。

    步骤：rdr2geo → geo2rdr → resample_slc
    输出：
      {work_dir}/scratch/coarse_resample_slc/freq{frequency}/{polarization}/
          coregistered_secondary.slc

    Parameters
    ----------
    reference, secondary : RSLC HDF5 路径（绝对路径）
    dem_file             : 本地 DEM（EPSG:4326，覆盖场景范围即可）
    work_dir             : 工作目录（scratch/ 将在此下创建）
    """
    from nisar.workflows import geo2rdr, rdr2geo, resample_slc

    scratch = str(Path(work_dir) / 'scratch')

    config = {
        'input_file_group': {
            'reference_rslc_file': str(Path(reference).resolve()),
            'secondary_rslc_file': str(Path(secondary).resolve()),
        },
        'dynamic_ancillary_file_group': {
            'dem_file': str(Path(dem_file).resolve()),
            'orbit_files': {
                'reference_orbit_file': None,
                'secondary_orbit_file': None,
            },
        },
        'product_path_group': {'scratch_path': scratch},
        'processing': {
            'rdr2geo': {
                'threshold': 1e-8,
                'numiter':   25,
                'extraiter': 10,
                'lines_per_block': BLOCK_SIZE,
                'write_x': True,
                'write_y': True,
                'write_z': True,
                'write_incidence':          False,
                'write_heading':            False,
                'write_local_incidence':    False,
                'write_local_psi':          False,
                'write_simulated_amplitude': False,
                'write_layover_shadow':     False,
            },
            'geo2rdr': {
                'threshold': 1e-8,
                'numiter':   25,
                'extraiter': 10,
                'lines_per_block': BLOCK_SIZE,
                'topo_path': scratch + '/',
                'maxiter': 10,
            },
            f'{resample_type}_resample': {
                'offsets_dir':   scratch + '/',
                'lines_per_tile': BLOCK_SIZE,
                'flatten':       False,
            },
            'input_subset': {
                'list_of_frequencies': {frequency: [polarization]},
            },
        },
        'worker': {'internet_access': False, 'gpu_enabled': False, 'gpu_id': 0},
    }

    log.info("ISCE3 rdr2geo ...")
    rdr2geo.run(config)
    log.info("ISCE3 geo2rdr ...")
    geo2rdr.run(config)
    log.info("ISCE3 resample_slc (%s) ...", resample_type)
    resample_slc.run(config, resample_type)
    log.info("ISCE3 协同配准完成。")


# ══════════════════════════════════════════════════════════════════════════════
# 5. RSLC 专用：Mock S1 Orbit File
#    ISCE3 loadMetadataRslc 需要 S1-格式轨道文件；从 HDF5 内嵌轨道生成
# ══════════════════════════════════════════════════════════════════════════════

def _mock_s1_orbit_file(h5_path: str) -> str:
    """从 NISAR HDF5 内嵌轨道生成 S1 格式 EOF 文件（完全对齐官方实现）。

    Returns
    -------
    orbit_path : str — 写出的 .EOF 文件路径（与 h5_path 同目录）
    """
    from numpy import datetime64, timedelta64
    from nisar.products.readers import product as nisar_product

    orbit_path = str(Path(h5_path).with_suffix('.EOF'))
    ds     = nisar_product.open_product(h5_path)
    orbit  = ds.getOrbit()
    count  = len(orbit.position)
    ref_epoch = datetime64(orbit.reference_epoch, 'ns')

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n',
        '<Earth_Explorer_File>\n',
        '\t<Data_Block type="xml">\n',
        f'\t\t<List_of_OSVs count="{count}">\n',
    ]

    for t, vel, pos in zip(orbit.time, orbit.velocity, orbit.position):
        utc = ref_epoch + timedelta64(int(t * 1e9), 'ns')
        lines += [
            '\t\t\t<OSV>\n',
            f'\t\t\t\t<UTC>UTC={utc}</UTC>\n',
            f'\t\t\t\t<X unit="m">{pos[0]}</X>\n',
            f'\t\t\t\t<Y unit="m">{pos[1]}</Y>\n',
            f'\t\t\t\t<Z unit="m">{pos[2]}</Z>\n',
            f'\t\t\t\t<VX unit="m/s">{vel[0]}</VX>\n',
            f'\t\t\t\t<VY unit="m/s">{vel[1]}</VY>\n',
            f'\t\t\t\t<VZ unit="m/s">{vel[2]}</VZ>\n',
            '\t\t\t</OSV>\n',
        ]

    lines += [
        '\t\t</List_of_OSVs>\n',
        '\t</Data_Block>\n',
        '</Earth_Explorer_File>\n',
    ]

    with open(orbit_path, 'w') as f:
        f.writelines(lines)

    log.info("Mock S1 orbit file 已写入: %s", orbit_path)
    return orbit_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. GSLC 专用：GSLCMetadata（Geogrid 接口对象）
# ══════════════════════════════════════════════════════════════════════════════

class GSLCMetadata:
    """GSLC 产品的 Geogrid 元数据，对齐官方 hyp3_autorift.nisar_isce3.GSLCMetadata。

    与 RSLC 的 loadMetadataRslc 返回对象接口相同，
    供 geogrid_builder.run_geogrid_sar 调用。
    """

    def __init__(self, tif_path: str, scene_name: str):
        self.filename = tif_path

        # ── 解析 sensingStart（从场景名提取 {YYYYMMDDTHHMMSS}）───────────
        parts    = Path(scene_name).stem.split('_')
        time_str = None
        for part in parts:
            if len(part) == 15 and 'T' in part:
                time_str = part
                break
        if time_str is None:
            # 回退：尝试位置 [11]（官方 NISAR 命名约定）
            try:
                time_str = parts[11]
            except IndexError:
                time_str = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
                log.warning(
                    "无法从场景名 '%s' 解析时间，使用当前 UTC 时间作为 sensingStart",
                    scene_name,
                )

        self.time         = time_str
        self.sensingStart = datetime.strptime(self.time, '%Y%m%dT%H%M%S')

        # ── 轨道参数（简化估算，官方使用 cycle / rel_orb）────────────────
        try:
            cycle   = int(parts[4])
            rel_orb = int(parts[5])
            self.absoluteOrbitNumber = 618 + cycle * 173 + rel_orb
        except (IndexError, ValueError):
            self.absoluteOrbitNumber = 0
            log.debug("无法解析 NISAR 轨道号，设为 0")

        self.orbitPassDirection = (
            'ASCENDING' if (len(parts) > 6 and parts[6] == 'A') else 'DESCENDING'
        )

        # ── 从输出 GeoTIFF 读取空间参数 ───────────────────────────────────
        ds = gdal.Open(tif_path)
        if ds is None:
            raise FileNotFoundError(f"GSLCMetadata: 无法打开 GeoTIFF: {tif_path}")
        trans = ds.GetGeoTransform()
        self.XSize             = trans[1]
        self.YSize             = trans[5]
        self.numberOfLines     = ds.RasterYSize
        self.numberOfSamples   = ds.RasterXSize
        self.startingX         = trans[0]
        self.startingY         = trans[3]
        ds = None


# ══════════════════════════════════════════════════════════════════════════════
# 7. 元数据对加载（统一接口）
# ══════════════════════════════════════════════════════════════════════════════

def load_nisar_metadata_pair(
    ref_h5:       str,
    sec_h5:       str,
    nisar_type:   str,
    ref_amp_tif:  str,
    sec_amp_tif:  str,
    frequency:    str = 'A',
    polarization: str = 'HH',
    orbit_path:   Optional[str] = None,
) -> Tuple[object, object, int]:
    """加载 NISAR 元数据对（适配 geogrid_builder.run_geogrid_sar 接口）。

    Returns
    -------
    meta_r, meta_s : Geogrid 元数据对象（与 S1 的 loadMetadataSlc 返回值接口一致）
    optical_flag   : 0（RSLC，斜距几何）/ 1（GSLC，地理编码）
    """
    if nisar_type == 'RSLC':
        from hyp3_autorift.vend.testGeogrid import loadMetadataRslc

        # 生成 mock S1 orbit file（RSLC 元数据加载所需）
        if orbit_path is None:
            orbit_path = _mock_s1_orbit_file(ref_h5)

        log.info("加载 RSLC 参考景元数据 ...")
        meta_r    = loadMetadataRslc(str(ref_h5), orbit_path=str(orbit_path))

        log.info("加载 RSLC 次景元数据（仅获取时间）...")
        meta_temp = loadMetadataRslc(str(sec_h5))

        meta_s = copy.copy(meta_r)
        meta_s.sensingStart = meta_temp.sensingStart
        meta_s.sensingStop  = meta_temp.sensingStop

        optical_flag = 0

    elif nisar_type == 'GSLC':
        log.info("构建 GSLC 元数据（GSLCMetadata）...")
        meta_r = GSLCMetadata(ref_amp_tif, Path(ref_h5).stem)
        meta_s = GSLCMetadata(sec_amp_tif, Path(sec_h5).stem)
        optical_flag = 1

    else:
        raise ValueError(f"不支持的 NISAR 类型: {nisar_type}")

    log.info(
        "元数据加载完成: ref_sensingStart=%s  optical_flag=%d",
        str(meta_r.sensingStart)[:19], optical_flag,
    )
    return meta_r, meta_s, optical_flag


# ══════════════════════════════════════════════════════════════════════════════
# 8. EPSG 辅助（本地管线使用 DEM/config 获取 EPSG，此函数为可选回退）
# ══════════════════════════════════════════════════════════════════════════════

def get_gslc_epsg(h5_path: str, frequency: str = 'A',
                  polarization: str = 'HH') -> Optional[int]:
    """从 GSLC HDF5 读取投影 EPSG（可选回退；RSLC 返回 None）。"""
    key = f'HDF5:{h5_path}://science/LSAR/GSLC/grids/frequency{frequency}/{polarization}'
    ds  = gdal.Open(key)
    if ds is None:
        return None
    srs = osr.SpatialReference(ds.GetProjection())
    srs.AutoIdentifyEPSG()
    code = srs.GetAuthorityCode(None)
    ds   = None
    return int(code) if code else None


# ══════════════════════════════════════════════════════════════════════════════
# 9. 主处理函数（供 run_pipeline.py 调用）
# ══════════════════════════════════════════════════════════════════════════════

def process_nisar(
    ref_h5:      str,
    sec_h5:      str,
    dem_file:    str,
    work_dir:    str,
    amp_ref_path: str,
    amp_sec_path: str,
    frequency:   str = 'A',
) -> Tuple[object, object, int]:
    """NISAR 完整预处理入口（供 run_pipeline.py 调用）。

    步骤
    ----
    1. 检测产品类型（RSLC / GSLC）
    2. 检测可用极化
    3. 产品处理：
       RSLC → ISCE3 协同配准 → 提取参考/次景幅度
       GSLC → 裁剪重叠区   → 提取参考/次景幅度
    4. 加载 Geogrid 元数据

    Parameters
    ----------
    ref_h5, sec_h5   : 本地 NISAR HDF5 路径
    dem_file         : 本地 DEM（RSLC 协同配准用；EPSG:4326 推荐）
    work_dir         : 工作目录（中间文件存放在此）
    amp_ref_path     : 参考幅度图输出路径（通常 results/{pair}/reference.tif）
    amp_sec_path     : 次景幅度图输出路径（通常 results/{pair}/secondary.tif）
    frequency        : NISAR 频段，默认 'A'

    Returns
    -------
    meta_r, meta_s : Geogrid 元数据对象
    optical_flag   : 0（RSLC）或 1（GSLC）
    """
    ref_h5    = str(Path(ref_h5).resolve())
    sec_h5    = str(Path(sec_h5).resolve())
    dem_file  = str(Path(dem_file).resolve())

    nisar_type   = detect_nisar_type(ref_h5)
    polarization = detect_polarization(ref_h5, frequency)

    log.info("=" * 60)
    log.info("NISAR 处理: type=%s  freq=%s  pol=%s", nisar_type, frequency, polarization)
    log.info("  参考: %s", ref_h5)
    log.info("  次景: %s", sec_h5)

    # ── 断点续算：幅度图已存在则跳过 ─────────────────────────────────────
    already_done = Path(amp_ref_path).exists() and Path(amp_sec_path).exists()

    if nisar_type == 'RSLC':
        if not already_done:
            # Step 3a: ISCE3 协同配准
            log.info("Step 3a: ISCE3 协同配准 ...")
            run_isce3_coreg(
                reference=ref_h5,
                secondary=sec_h5,
                dem_file=dem_file,
                work_dir=work_dir,
                frequency=frequency,
                polarization=polarization,
            )

            # Step 3b: 提取参考景幅度
            log.info("Step 3b: 提取参考景幅度 ...")
            ref_data = (
                f'HDF5:{ref_h5}://science/LSAR/RSLC/'
                f'swaths/frequency{frequency}/{polarization}'
            )
            extract_amplitude(ref_data, amp_ref_path, is_geocoded=False)

            # Step 3c: 提取协同配准后次景幅度
            log.info("Step 3c: 提取次景幅度（协同配准后）...")
            sec_coreg = str(
                Path(work_dir) / 'scratch' / 'coarse_resample_slc'
                / f'freq{frequency}' / polarization
                / 'coregistered_secondary.slc'
            )
            if not Path(sec_coreg).exists():
                raise FileNotFoundError(
                    f"ISCE3 协同配准输出未找到: {sec_coreg}\n"
                    "请检查 run_isce3_coreg 是否正常完成。"
                )
            extract_amplitude(sec_coreg, amp_sec_path, is_geocoded=False)
        else:
            log.info("幅度图已存在，跳过 ISCE3 & 幅度提取（断点续算）。")

    elif nisar_type == 'GSLC':
        if not already_done:
            # Step 3a: 裁剪至重叠区
            log.info("Step 3a: GSLC 裁剪至空间重叠区 ...")
            ref_cropped, sec_cropped = crop_gslcs_overlap(
                ref_h5, sec_h5, work_dir, frequency, polarization
            )

            # Step 3b/3c: 提取幅度（保留地理坐标）
            log.info("Step 3b: 提取参考/次景幅度（GSLC，携带地理坐标）...")
            extract_amplitude(ref_cropped, amp_ref_path, is_geocoded=True)
            extract_amplitude(sec_cropped, amp_sec_path, is_geocoded=True)
        else:
            log.info("幅度图已存在，跳过裁剪 & 幅度提取（断点续算）。")

    else:
        raise ValueError(f"不支持的 NISAR 类型: {nisar_type}")

    # Step 4: 加载 Geogrid 元数据
    log.info("Step 4: 加载 NISAR Geogrid 元数据 ...")
    meta_r, meta_s, optical_flag = load_nisar_metadata_pair(
        ref_h5=ref_h5,
        sec_h5=sec_h5,
        nisar_type=nisar_type,
        ref_amp_tif=amp_ref_path,
        sec_amp_tif=amp_sec_path,
        frequency=frequency,
        polarization=polarization,
    )

    return meta_r, meta_s, optical_flag
