#!/usr/bin/env python3
"""
batch_run.py — 批量运行 autoRIFT_Optical_Local_Pipeline（SBAS 网络 + 跨传感器 + 可选波段）

平台族（--platform）：
  landsat     L4/L5/L7/L8/L9 混合（默认波段：B8 全色 / B2 绿波段）
  s2          Sentinel-2A/2B（默认波段：B08 NIR）
  planetscope PlanetScope PSScene（默认波段：*_SR_*.tif）
  custom      自定义（必须指定 --bands）

波段选择（--bands）：
  覆盖各平台默认波段，支持一个或多个文件名后缀 / glob pattern。
  示例：
    --bands B08                    → 搜索 *B08.jp2 和 *B08.tif
    --bands B04 B08                → 同时搜索 B04 和 B08（多波段，取第一个找到的）
    --bands "*_SR_3B_*.tif"        → 直接指定 glob（PlanetScope 3波段）
    --bands "*_B3.TIF" "*_B4.TIF"  → 多个 glob

用法：
  # Landsat 默认波段（B8全色）
  python batch_run.py --scene-dir /data1/LS --output-dir ./results \
    --platform landsat --interval 120 --dry-run

  # Landsat 强制使用 B4（红波段）
  python batch_run.py --scene-dir /data1/LS --output-dir ./results \
    --platform landsat --interval 120 --bands B4 --dry-run

  # S2 改用 B04（红波段）
  python batch_run.py --scene-dir /data1/S2 --output-dir ./results \
    --platform s2 --interval 60 --bands B04 --dry-run

  # S2 同时搜索 B08 和 B04（多波段，优先 B08）
  python batch_run.py --scene-dir /data1/S2 --output-dir ./results \
    --platform s2 --interval 60 --bands B08 B04 --dry-run

  # PlanetScope 指定 3波段 SR
  python batch_run.py --scene-dir /data1/PS --output-dir ./results \
    --platform planetscope --interval 30 --bands "*_SR_3B_*.tif" --dry-run

  # 完全自定义（SPOT 全色）
  python batch_run.py --scene-dir /data1/SPOT --output-dir ./results \
    --platform custom --bands "*_P.tif" --date-regex "_(\d{8})_" \
    --interval 60 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 空间重叠度计算
# ─────────────────────────────────────────────────────────────────────────────

# 每个路径只打开一次，后续从缓存读取
_BBOX_CACHE: Dict[str, Tuple[float, float, float, float]] = {}


def _get_raster_bbox_wgs84(path: str) -> Tuple[float, float, float, float]:
    """从栅格文件读取包围框，转换为 WGS84 (lon_min, lat_min, lon_max, lat_max)。

    结果缓存到 _BBOX_CACHE，同路径只读一次。
    """
    if path in _BBOX_CACHE:
        return _BBOX_CACHE[path]

    from osgeo import gdal, osr
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开: {path}")
    gt   = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    ds = None

    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + gt[1] * cols
    y_min = y_max + gt[5] * rows  # gt[5] < 0

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    # GDAL 3.x 默认经纬度轴序可能为 (lat, lon)，强制 (lon, lat)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if src_srs.IsSame(wgs84):
        result: Tuple[float, float, float, float] = (x_min, y_min, x_max, y_max)
    else:
        tr = osr.CoordinateTransformation(src_srs, wgs84)
        corners = [
            tr.TransformPoint(x_min, y_min),
            tr.TransformPoint(x_min, y_max),
            tr.TransformPoint(x_max, y_min),
            tr.TransformPoint(x_max, y_max),
        ]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        result = (min(lons), min(lats), max(lons), max(lats))

    _BBOX_CACHE[path] = result
    return result


def compute_overlap_fraction(path_a: str, path_b: str) -> float:
    """计算两景影像包围框的空间重叠度（相对于较小景面积，返回 0.0–1.0）。

    若无法读取则返回 1.0（保守：视为完全重叠，不过滤）。
    """
    try:
        ax1, ay1, ax2, ay2 = _get_raster_bbox_wgs84(path_a)
        bx1, by1, bx2, by2 = _get_raster_bbox_wgs84(path_b)
    except Exception as e:
        log.debug("重叠度计算失败（%s / %s）: %s — 保守处理视为完全重叠",
                  Path(path_a).name, Path(path_b).name, e)
        return 1.0

    # 交叉包围框
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    return min(inter / min_area, 1.0)


def filter_pairs_by_overlap(
    pairs: List[Tuple[Dict, Dict]],
    min_overlap: float = 0.20,
) -> Tuple[List[Tuple[Dict, Dict]], Dict[str, float]]:
    """按空间重叠度过滤影像对。

    Parameters
    ----------
    pairs       : build_pairs_sbas 返回的配对列表
    min_overlap : 最小重叠度阈值（0–1），0% 的影像对始终剔除

    Returns
    -------
    filtered    : 通过筛选的配对列表
    overlap_map : {pair_name: overlap_fraction}（含所有原始对，用于摘要显示）
    """
    overlap_map: Dict[str, float] = {}
    filtered: List[Tuple[Dict, Dict]] = []
    n_zero = n_low = 0

    log.info("计算 %d 个配对的空间重叠度（阈值 %.0f%%）...", len(pairs), min_overlap * 100)
    for ref, sec in pairs:
        pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
        frac = compute_overlap_fraction(ref["path"], sec["path"])
        overlap_map[pair_name] = frac

        if frac == 0.0:
            n_zero += 1
            log.debug("  [剔除-零重叠] %s  overlap=0%%", pair_name)
        elif frac < min_overlap:
            n_low += 1
            log.debug("  [剔除-低重叠] %s  overlap=%.1f%% < %.0f%%",
                      pair_name, frac * 100, min_overlap * 100)
        else:
            filtered.append((ref, sec))

    log.info(
        "重叠度过滤: %d 对通过 | %d 对零重叠剔除 | %d 对低于阈值剔除",
        len(filtered), n_zero, n_low,
    )
    return filtered, overlap_map


# ─────────────────────────────────────────────────────────────────────────────
# 平台族定义
# ─────────────────────────────────────────────────────────────────────────────
# default_bands: 每个平台族的内置默认波段标识符列表（可被 --bands 覆盖）
# 每个标识符会被 _bands_to_globs() 展开为实际的 glob pattern 列表

PLATFORM_FAMILIES: Dict[str, Dict] = {
    "landsat": {
        # 默认：B8 全色（L7/8/9）+ B2 绿波段（L4/5）
        "default_bands": ["B8", "B2"],
        "date_patterns": [
            re.compile(r"_(\d{8})_\d{8}_\d{2}_T"),  # 标准 Collection 2
            re.compile(r"_(\d{8})_"),                  # 宽松回退
        ],
        "sensor_patterns": {
            "L9": re.compile(r"^L[CO]09_", re.I),
            "L8": re.compile(r"^L[CO]08_", re.I),
            "L7": re.compile(r"^L[EO]07_", re.I),
            "L5": re.compile(r"^LT05_",    re.I),
            "L4": re.compile(r"^LT04_",    re.I),
        },
        "config_platform_map": {
            "L9": "L9", "L8": "L8", "L7": "L7", "L5": "L5", "L4": "L4",
        },
    },

    "s2": {
        # 默认：B08 NIR（10m）
        "default_bands": ["B08"],
        "date_patterns": [
            re.compile(r"_(\d{8})T\d{6}_"),  # ESA
            re.compile(r"_(\d{8})_"),          # COG
        ],
        "sensor_patterns": {
            "S2A": re.compile(r"^S2A_", re.I),
            "S2B": re.compile(r"^S2B_", re.I),
        },
        "config_platform_map": {"S2A": "S2", "S2B": "S2"},
    },

    "planetscope": {
        # 默认：分析就绪 SR 影像（多种命名方式）
        "default_bands": ["*_SR_*.tif", "*_AnalyticMS_SR_*.tif", "*_Analytic_*.tif"],
        "date_patterns": [
            re.compile(r"^(\d{8})_"),   # YYYYMMDD_HHMMSS_...
            re.compile(r"_(\d{8})_"),
        ],
        "sensor_patterns": {
            "PS2":    re.compile(r"_PS2_",    re.I),
            "PS2.SD": re.compile(r"_PS2\.SD_", re.I),
            "PSB.SD": re.compile(r"_PSB\.SD_", re.I),
        },
        "config_platform_map": {},
    },

    "custom": {
        # 必须由 --bands 指定
        "default_bands": [],
        "date_patterns": [],   # 由 --date-regex 指定
        "sensor_patterns": {},
        "config_platform_map": {},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 波段标识符 → glob pattern 列表
# ─────────────────────────────────────────────────────────────────────────────

def _bands_to_globs(bands: List[str]) -> List[str]:
    """将波段标识符列表展开为文件搜索 glob pattern 列表。

    规则：
      - 若标识符本身包含 '*' 或 '.' → 视为完整 glob，直接使用
      - 否则视为波段名（如 "B08"、"B8"、"B4"），自动生成常见后缀组合：
          *{BAND}.jp2  *{BAND}.tif  *{BAND}.TIF  *_{BAND}.jp2  *_{BAND}.tif  *_{BAND}.TIF

    示例：
      ["B08"]            → ["*B08.jp2", "*B08.tif", "*B08.TIF",
                             "*_B08.jp2", "*_B08.tif", "*_B08.TIF"]
      ["B08", "B04"]     → 以上 × 2（两组合并）
      ["*_SR_3B_*.tif"]  → ["*_SR_3B_*.tif"]（直接透传）
    """
    globs = []
    for band in bands:
        if "*" in band or "." in band:
            # 视为完整 glob pattern，直接使用
            globs.append(band)
        else:
            # 视为波段名，展开为多种后缀
            b = band  # e.g. "B08" / "B8" / "B4"
            globs.extend([
                f"*{b}.jp2",
                f"*{b}.tif",
                f"*{b}.TIF",
                f"*_{b}.jp2",
                f"*_{b}.tif",
                f"*_{b}.TIF",
            ])
    # 去重，保持顺序
    seen = set()
    unique = []
    for g in globs:
        if g not in seen:
            unique.append(g)
            seen.add(g)
    return unique


def _globs_to_suffixes(globs: List[str]) -> List[str]:
    """从 glob pattern 提取用于场景名截断的文件名后缀。

    只处理不含通配符的尾部（*_B08.tif → _B08.tif）；
    含中间通配符的 glob（如 *_SR_*.tif）无法静态提取，返回空串（场景名回退到 stem）。
    """
    suffixes = []
    for g in globs:
        # 去掉前导 * 和 *_，得到可能的固定后缀
        candidate = re.sub(r"^\*_?", "", g)
        if "*" not in candidate and "?" not in candidate:
            suffixes.append(candidate)
            # 同时尝试大小写变体
            if candidate != candidate.upper():
                suffixes.append(candidate.upper())
            if candidate != candidate.lower():
                suffixes.append(candidate.lower())
    return list(dict.fromkeys(suffixes))  # 去重保序


# ─────────────────────────────────────────────────────────────────────────────
# 场景扫描
# ─────────────────────────────────────────────────────────────────────────────

def parse_scene_dir(
    scene_dir: Path,
    platform_family: str,
    bands: Optional[List[str]] = None,
    custom_date_regex: Optional[str] = None,
) -> List[Dict]:
    """扫描场景目录，返回按日期排序的影像记录列表（去重）。

    Parameters
    ----------
    bands             : 波段标识符列表（覆盖平台默认值）。
                        None → 使用平台默认 default_bands。
    custom_date_regex : custom 模式日期正则（含一个捕获组）。
    """
    fam = PLATFORM_FAMILIES[platform_family]

    # 确定有效波段
    effective_bands = bands if bands else fam["default_bands"]
    if not effective_bands:
        raise ValueError(
            f"平台族 '{platform_family}' 没有默认波段，请通过 --bands 指定。"
        )

    globs    = _bands_to_globs(effective_bands)
    suffixes = _globs_to_suffixes(globs)
    log.info(
        "波段配置: %s  →  globs: %s",
        effective_bands,
        globs if len(globs) <= 4 else globs[:4] + ["..."],
    )

    # 日期正则
    date_patterns = fam["date_patterns"]
    if platform_family == "custom" and custom_date_regex:
        date_patterns = [re.compile(custom_date_regex)]
    if not date_patterns:
        raise ValueError(
            f"平台族 '{platform_family}' 无日期解析规则，请通过 --date-regex 指定。"
        )

    # 收集文件
    found_files: List[Path] = []
    for pattern in globs:
        found_files.extend(scene_dir.rglob(pattern))
    found_files = sorted(set(found_files))
    log.debug("候选文件数: %d", len(found_files))

    records = []
    for f in found_files:
        try:
            date_str    = _extract_date(f.name, date_patterns)
            sensor      = _detect_sensor(f.name, fam["sensor_patterns"])
            scene_name  = _extract_scene_name(f.name, suffixes)
            config_plat = fam["config_platform_map"].get(sensor, platform_family.upper())
            records.append({
                "path":            str(f.resolve()),
                "scene_name":      scene_name,
                "date":            datetime.strptime(date_str, "%Y%m%d"),
                "sensor":          sensor,
                "family":          platform_family,
                "config_platform": config_plat,
                "band":            _guess_band(f.name, effective_bands),
            })
        except Exception as e:
            log.debug("跳过 %s: %s", f.name, e)

    # 排序 + 去重（同日期同场景名取第一个波段）
    records.sort(key=lambda r: (r["date"], r["scene_name"], r["band"]))
    seen, unique = set(), []
    for r in records:
        key = (r["date"], r["scene_name"])
        if key not in seen:
            unique.append(r)
            seen.add(key)

    # 统计
    sensor_count: Dict[str, int] = {}
    band_count:   Dict[str, int] = {}
    for r in unique:
        sensor_count[r["sensor"]] = sensor_count.get(r["sensor"], 0) + 1
        band_count[r["band"]]     = band_count.get(r["band"], 0) + 1

    sensor_s = "  ".join(f"{k}×{v}" for k, v in sorted(sensor_count.items()))
    band_s   = "  ".join(f"{k}×{v}" for k, v in sorted(band_count.items()))
    log.info("扫描到 %d 个有效场景（%s）", len(unique), platform_family)
    log.info("  传感器: %s", sensor_s)
    log.info("  波段:   %s", band_s)
    return unique


def _extract_date(name: str, patterns: List[re.Pattern]) -> str:
    for pat in patterns:
        m = pat.search(name)
        if m:
            return m.group(1)
    raise ValueError(f"无法从 '{name}' 提取日期")


def _detect_sensor(name: str, sensor_patterns: Dict[str, re.Pattern]) -> str:
    for sensor, pat in sensor_patterns.items():
        if pat.search(name):
            return sensor
    return "UNKNOWN"


def _extract_scene_name(name: str, suffixes: List[str]) -> str:
    """截去已知波段后缀，得到场景基础名。"""
    for suffix in suffixes:
        if name.upper().endswith(suffix.upper()):
            return name[: -len(suffix)]
    return Path(name).stem


def _guess_band(name: str, bands: List[str]) -> str:
    """尝试从文件名反推使用的波段标识符。"""
    for b in bands:
        # 去掉 glob 通配符，用剩余部分做简单匹配
        clean = b.replace("*", "").replace("?", "").strip("_.")
        if clean and clean.upper() in name.upper():
            return b
    return bands[0] if bands else "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# SBAS 网络配对
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs_sbas(
    records: List[Dict],
    max_days: int,
    min_days: int = 1,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
    exclude_same_sensor: bool = False,
    require_same_sensor: bool = False,
) -> List[Tuple[Dict, Dict]]:
    """枚举所有时间基线在 [min_days, max_days] 内的影像对。"""
    if date_start:
        records = [r for r in records if r["date"] >= date_start]
    if date_end:
        records = [r for r in records if r["date"] <= date_end]

    pairs = []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            dt = (records[j]["date"] - records[i]["date"]).days
            if dt < min_days:
                continue
            if dt > max_days:
                break
            ref, sec = records[i], records[j]
            if require_same_sensor and ref["sensor"] != sec["sensor"]:
                continue
            if exclude_same_sensor and ref["sensor"] == sec["sensor"]:
                continue
            pairs.append((ref, sec))

    if pairs:
        spans = [(sec["date"] - ref["date"]).days for ref, sec in pairs]
        cross = sum(1 for ref, sec in pairs if ref["sensor"] != sec["sensor"])
        log.info(
            "SBAS 网络：%d 对  基线 %d–%d 天（均值 %.1f）  跨传感器 %d 对",
            len(pairs), min(spans), max(spans), sum(spans) / len(spans), cross,
        )
    else:
        log.warning("未生成任何配对（max_days=%d，场景数=%d）", max_days, len(records))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 网络可视化
# ─────────────────────────────────────────────────────────────────────────────

def print_network_summary(
    records: List[Dict],
    pairs: List[Tuple[Dict, Dict]],
    output_dir: str,
    bands: List[str],
    overlap_map: Optional[Dict[str, float]] = None,
) -> None:
    from collections import Counter

    conn: Counter = Counter()
    sensor_pairs: Counter = Counter()
    for ref, sec in pairs:
        conn[ref["date"]] += 1
        conn[sec["date"]] += 1
        combo = (f"{ref['sensor']}+{sec['sensor']}"
                 if ref["sensor"] != sec["sensor"] else ref["sensor"])
        sensor_pairs[combo] += 1

    print(f"\n{'='*80}")
    print(f"  SBAS 网络：{len(records)} 景 → {len(pairs)} 个配对")
    print(f"  使用波段: {bands}")
    print(f"  传感器组合: " + "  ".join(f"{k}×{v}" for k, v in sensor_pairs.most_common()))
    print(f"{'='*80}")

    print(f"\n  {'日期':^12}  {'传感器':^7}  {'波段':^10}  {'连接':^4}  图")
    print(f"  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*4}  {'-'*20}")
    for r in records:
        d = r["date"]
        c = conn.get(d, 0)
        band_label = r.get("band", "")[:10]
        print(f"  {d:%Y-%m-%d}  {r['sensor']:^7}  {band_label:^10}  {c:^4}  {'█' * min(c, 20)}")

    show_overlap = overlap_map is not None
    if show_overlap:
        print(f"\n  {'#':>4}  {'参考景':^12}  {'传感器':^7}  {'次景':^12}  {'传感器':^7}  {'基线(天)':>8}  {'重叠%':>6}  状态")
        print(f"  {'-'*4}  {'-'*12}  {'-'*7}  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*8}")
    else:
        print(f"\n  {'#':>4}  {'参考景':^12}  {'传感器':^7}  {'次景':^12}  {'传感器':^7}  {'基线(天)':>8}  状态")
        print(f"  {'-'*4}  {'-'*12}  {'-'*7}  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*8}")

    for i, (ref, sec) in enumerate(pairs, 1):
        span = (sec["date"] - ref["date"]).days
        pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
        status = "✓ 已完成" if _is_completed(output_dir, pair_name) else "○ 待处理"
        cross  = " ←跨" if ref["sensor"] != sec["sensor"] else ""
        if show_overlap:
            ov = overlap_map.get(pair_name, float("nan"))
            ov_str = f"{ov*100:5.1f}%" if ov == ov else "  N/A "
            print(
                f"  {i:4d}  {ref['date']:%Y-%m-%d}  {ref['sensor']:^7}  "
                f"{sec['date']:%Y-%m-%d}  {sec['sensor']:^7}  {span:>8}  {ov_str}  {status}{cross}"
            )
        else:
            print(
                f"  {i:4d}  {ref['date']:%Y-%m-%d}  {ref['sensor']:^7}  "
                f"{sec['date']:%Y-%m-%d}  {sec['sensor']:^7}  {span:>8}  {status}{cross}"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 配置生成 & 执行
# ─────────────────────────────────────────────────────────────────────────────

def _generate_config(
    ref: Dict,
    sec: Dict,
    base_config: str,
    output_dir: str,
    pair_name: str,
) -> str:
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["input"]["reference_path"]  = ref["path"]
    cfg["input"]["secondary_path"]  = sec["path"]
    cfg["input"]["reference_scene"] = ref["scene_name"]
    cfg["input"]["secondary_scene"] = sec["scene_name"]
    cfg["input"]["platform"]        = ref["config_platform"]
    cfg["processing"]["pair_name"]  = pair_name

    pair_dir = Path(output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    config_path = pair_dir / f"{pair_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return str(config_path)


def _is_completed(output_dir: str, pair_name: str) -> bool:
    return any((Path(output_dir) / pair_name).glob("velocity_*.tif"))


def _run_pair(
    ref: Dict,
    sec: Dict,
    args: argparse.Namespace,
    base_config: str,
) -> Tuple[str, bool, Optional[str]]:
    pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
    if _is_completed(args.output_dir, pair_name):
        return pair_name, True, None

    try:
        config_path = _generate_config(
            ref=ref, sec=sec,
            base_config=base_config,
            output_dir=args.output_dir,
            pair_name=pair_name,
        )
        cmd = [sys.executable,
               str(Path(args.pipeline_dir) / "run_pipeline.py"),
               "--config", config_path]
        if args.verbose:
            cmd.append("--verbose")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        log.info("[%s] %s+%s 开始...", pair_name, ref["sensor"], sec["sensor"])
        with subprocess.Popen(
            cmd, cwd=str(args.pipeline_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        ) as proc:
            for line in proc.stdout:
                print(f"[{pair_name}] {line.rstrip()}", flush=True)
            proc.wait()

        if proc.returncode != 0:
            return pair_name, False, f"退出码 {proc.returncode}"
        log.info("[%s] ✓ 成功", pair_name)
        return pair_name, True, None

    except Exception as e:
        return pair_name, False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="批量运行 autoRIFT_Optical_Local_Pipeline（SBAS + 跨传感器 + 可选波段）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--scene-dir",  required=True, help="本地场景目录")
    p.add_argument("--output-dir", required=True, help="结果输出根目录")
    p.add_argument(
        "--platform", required=True,
        choices=["landsat", "s2", "planetscope", "custom"],
        help=(
            "平台族：\n"
            "  landsat     — L4/L5/L7/L8/L9 混合（默认波段：B8/B2）\n"
            "  s2          — Sentinel-2A/2B（默认波段：B08）\n"
            "  planetscope — PlanetScope PSScene（默认：*_SR_*.tif）\n"
            "  custom      — 自定义（必须指定 --bands 和 --date-regex）"
        ),
    )

    # ── 波段选择 ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--bands", nargs="+", metavar="BAND_OR_GLOB",
        help=(
            "覆盖平台默认波段，支持一个或多个：\n"
            "  波段名（自动展开）: B08  B04  B8  B3\n"
            "  完整 glob（含*或.）: '*_SR_3B_*.tif'  '*_B3.TIF'\n"
            "  多波段（同场景取第一个找到的）: B08 B04\n"
            "默认（不指定）: 使用平台内置默认波段"
        ),
    )

    # ── custom 模式 ───────────────────────────────────────────────────────────
    p.add_argument(
        "--date-regex", metavar="REGEX",
        help="[custom/补充] 从文件名提取 YYYYMMDD 的正则（一个捕获组）。"
             "例：'_(\\d{8})_'",
    )

    p.add_argument("--pipeline-dir", default=".",           help="pipeline 根目录")
    p.add_argument("--base-config",  default="config.yaml", help="基础配置文件路径")

    # ── SBAS ──────────────────────────────────────────────────────────────────
    p.add_argument("--interval", type=int, default=120,
                   help="最大时间基线（天）：生成所有间隔 ≤ 该值的配对")
    p.add_argument("--min-days", type=int, default=1,
                   help="最小时间基线（天）")

    # ── 传感器过滤（互斥）────────────────────────────────────────────────────
    sg = p.add_mutually_exclusive_group()
    sg.add_argument("--cross-sensor-only", action="store_true",
                    help="只保留跨传感器对（如 L9+L8）")
    sg.add_argument("--same-sensor-only",  action="store_true",
                    help="只保留同传感器对")

    p.add_argument("--date-start",      help="起始日期 YYYY-MM-DD")
    p.add_argument("--date-end",        help="结束日期 YYYY-MM-DD")
    p.add_argument(
        "--min-overlap", type=float, default=0.20, metavar="FRAC",
        help="最小空间重叠度阈值（0–1）。重叠度为 0 的影像对始终剔除；"
             "低于此阈值的影像对跳过不处理（默认 0.20 = 20%%）",
    )
    p.add_argument("--dry-run",         action="store_true", help="仅打印网络结构，不执行")
    p.add_argument("--max-workers",     type=int, default=1, help="并行进程数（1=串行）")
    p.add_argument("--timeout-minutes", type=int, default=120)
    p.add_argument("--log-file",        help="日志文件路径")
    p.add_argument("--verbose", "-v",   action="store_true")

    args = p.parse_args()

    # 校验 custom
    if args.platform == "custom" and not args.bands:
        p.error("--platform custom 必须同时指定 --bands")

    # ── 日志 ──────────────────────────────────────────────────────────────────
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # ── 路径校验 ──────────────────────────────────────────────────────────────
    scene_dir    = Path(args.scene_dir)
    output_dir   = Path(args.output_dir)
    pipeline_dir = Path(args.pipeline_dir)
    base_config  = Path(args.base_config)

    for path, name in [
        (scene_dir,   "scene-dir"),
        (base_config, "base-config"),
        (pipeline_dir / "run_pipeline.py", "run_pipeline.py"),
    ]:
        if not path.exists():
            log.error("%s 不存在: %s", name, path)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 确定有效波段（用于日志/摘要显示）────────────────────────────────────
    effective_bands = args.bands or PLATFORM_FAMILIES[args.platform]["default_bands"]

    # ── 扫描场景 ──────────────────────────────────────────────────────────────
    records = parse_scene_dir(
        scene_dir, args.platform,
        bands=args.bands,
        custom_date_regex=args.date_regex,
    )
    if not records:
        log.error("未找到有效场景（platform=%s，bands=%s）", args.platform, effective_bands)
        sys.exit(1)

    date_start = datetime.strptime(args.date_start, "%Y-%m-%d") if args.date_start else None
    date_end   = datetime.strptime(args.date_end,   "%Y-%m-%d") if args.date_end   else None

    # ── SBAS 配对 ─────────────────────────────────────────────────────────────
    pairs = build_pairs_sbas(
        records,
        max_days=args.interval,
        min_days=args.min_days,
        date_start=date_start,
        date_end=date_end,
        exclude_same_sensor=args.cross_sensor_only,
        require_same_sensor=args.same_sensor_only,
    )
    if not pairs:
        log.error("未生成任何配对")
        sys.exit(1)

    # ── 空间重叠度过滤 ────────────────────────────────────────────────────────
    if args.min_overlap > 0 or True:   # 始终剔除零重叠对
        pairs, overlap_map = filter_pairs_by_overlap(pairs, min_overlap=args.min_overlap)
    else:
        overlap_map = None

    if not pairs:
        log.error("重叠度过滤后无剩余配对（--min-overlap=%.2f）", args.min_overlap)
        sys.exit(1)

    print_network_summary(records, pairs, str(output_dir), effective_bands,
                          overlap_map=overlap_map)

    if args.dry_run:
        log.info("干运行模式，退出")
        return

    # ── 执行 ──────────────────────────────────────────────────────────────────
    log.info("开始批量处理（并行度=%d，共 %d 对）", args.max_workers, len(pairs))
    ok, fail, skip = 0, 0, 0
    failed_pairs = []

    if args.max_workers == 1:
        for ref, sec in pairs:
            pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
            if _is_completed(str(output_dir), pair_name):
                log.info("[%s] 已完成，跳过", pair_name)
                skip += 1
                continue
            pname, success, err = _run_pair(ref, sec, args, str(base_config))
            if success:
                ok += 1
            else:
                log.error("[%s] ✗ 失败: %s", pname, err)
                fail += 1
                failed_pairs.append((pname, err))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_run_pair, ref, sec, args, str(base_config)): (ref, sec)
                for ref, sec in pairs
            }
            for fut in as_completed(futures):
                ref, sec = futures[fut]
                pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
                try:
                    pname, success, err = fut.result()
                    if success:
                        ok += 1
                    else:
                        log.error("[%s] ✗ 失败: %s", pname, err)
                        fail += 1
                        failed_pairs.append((pname, err))
                except Exception as e:
                    log.error("[%s] ✗ 异常: %s", pair_name, e)
                    fail += 1
                    failed_pairs.append((pair_name, str(e)))

    print(f"\n{'='*60}")
    print(f"批量完成：{ok} 成功 | {fail} 失败 | {skip} 已跳过")
    if failed_pairs:
        print("\n失败配对：")
        for pname, err in failed_pairs:
            print(f"  - {pname}: {err[:100]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()