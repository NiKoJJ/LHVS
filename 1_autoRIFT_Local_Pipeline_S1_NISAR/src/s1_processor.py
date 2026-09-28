"""
s1_processor.py — Sentinel-1 多 Swath SLC → CSLC 幅度图（完整对齐版）

【接口保持】函数名与参数与您的 run_pipeline.py 完全兼容
【逻辑对齐】集成 hyp3_autoRIFT 官方核心：
  1. 共视 Burst 交集计算（支持跨卫星/跨轨道容错）
  2. 次景 scratch_folder = product_folder（确保 .off 文件路径一致）
  3. 重叠区中点切分 + 非末端 Swath 64px 缓冲（消除 seam 伪影）
  4. 目录结构对齐：product/, product_sec/, scratch/, s1_cslc.yaml, reference.tif, secondary.tif

【本地化】完全移除 S3 依赖，强制本地 rdr2geo，路径全绝对化
"""
import copy
import glob
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
log = logging.getLogger(__name__)

YAML_TEMPLATE = Path(__file__).parent / "schemas" / "s1_cslc_template.yaml"


# ── 1. 共视 Burst 交集计算（新增，官方逻辑）─────────────────────────────────
def get_common_burst_ids(
    ref_zip: str, sec_zip: str, orbit_ref: str, orbit_sec: str
) -> Tuple[List[str], List[int]]:
    """计算参考景与次景的共视 Burst 交集（官方 list(set(sec) & set(ref)) 本地化）"""
    import s1reader

    def _extract(zip_path: str, orbit: str) -> List[Dict]:
        records = []
        for swath in [1, 2, 3]:
            for pol in ["vv", "vh", "hh", "hv"]:
                try:
                    bursts = s1reader.load_bursts(zip_path, orbit, swath, pol)
                    if bursts:
                        for b in bursts:
                            bid = f"t{int(b.burst_id.track_number):03d}_{str(b.burst_id.esa_burst_id).zfill(6)}_{b.burst_id.subswath.lower()}"
                            records.append({"id": bid, "swath": swath, "pol": pol})
                        break
                except Exception:
                    continue
        return records

    ref_ids = _extract(ref_zip, orbit_ref)
    sec_ids = _extract(sec_zip, orbit_sec)

    ref_set = {r["id"] for r in ref_ids}
    sec_set = {r["id"] for r in sec_ids}
    common = sorted(list(ref_set & sec_set))

    if not common:
        raise ValueError("参考景与次景无共视 Burst，请检查轨道或覆盖范围")
    if len(common) < len(ref_set):
        log.warning(
            "Burst 未完全匹配: 参考景 %d / 次景 %d / 共视 %d，仅处理重叠区域",
            len(ref_set), len(sec_set), len(common),
        )

    swaths_used = sorted(list({r["swath"] for r in ref_ids if r["id"] in common}))
    return common, swaths_used


# ── 2. 列出所有 burst（保持原接口）─────────────────────────────────────────
def get_all_bursts(zip_path: str, orbit_file: Optional[str] = None) -> List[Dict]:
    """返回 zip 中 IW1/IW2/IW3 所有 burst 的信息列表。"""
    import s1reader

    abs_zip = str(Path(zip_path).resolve())
    records = []

    for swath in [1, 2, 3]:
        pol = _detect_pol(abs_zip, orbit_file or "", swath)
        if pol is None:
            log.debug("IW%d: 无可用极化，跳过", swath)
            continue
        try:
            bursts = s1reader.load_bursts(abs_zip, orbit_file or "", swath, pol)
        except Exception as e:
            log.debug("IW%d 读取失败: %s", swath, e)
            continue

        for idx, burst in enumerate(bursts):
            isce3_id = _burst_to_isce3_id(burst)
            records.append({
                "swath":         swath,
                "burst_index":   idx,
                "isce3_id":      isce3_id,
                "polarization":  pol,
                "sensing_start": str(burst.sensing_start),
                "burst_obj":     burst,
            })

    return records


def _detect_pol(abs_zip: str, orbit_file: str, swath: int) -> Optional[str]:
    """探测可用极化（VV > VH > HH > HV）。"""
    import s1reader
    for pol in ["vv", "vh", "hh", "hv"]:
        try:
            if s1reader.load_bursts(abs_zip, orbit_file, swath, pol):
                return pol
        except Exception:
            continue
    return None


def _burst_to_isce3_id(burst) -> str:
    """burst → t{track:03d}_{esa_id:06d}_{subswath}"""
    track   = int(burst.burst_id.track_number)
    esa_id  = int(burst.burst_id.esa_burst_id)
    subswath = burst.burst_id.subswath.lower()
    return f"t{track:03d}_{esa_id:06d}_{subswath}"


# ── 3. 轨道文件（保持原接口）───────────────────────────────────────────────
def get_orbit_file(
    zip_path: str,
    orbit_dir: str,
    allow_download: bool = True,
) -> str:
    """查找本地轨道文件，找不到时下载（或报错）。"""
    from s1reader.s1_orbit import get_orbit_file_from_dir, retrieve_orbit_file

    orbit_dir_path = Path(orbit_dir)
    orbit_dir_path.mkdir(parents=True, exist_ok=True)

    log.info("search for %s orbit EOF ...", Path(zip_path).name)

    try:
        orbit_file = get_orbit_file_from_dir(
            zip_path, 
            str(orbit_dir_path), 
            auto_download=False  # 先尝试本地
        )
        if orbit_file:
            log.info("found in local: %s", orbit_file)
            return orbit_file
    except Exception as e:
        log.debug("本地未找到，尝试下载: %s", e)

    if allow_download:
        abs_zip = str(Path(zip_path).resolve())
        log.info("正在下载轨道文件: %s → %s", Path(zip_path).name, orbit_dir)
        orbit_file = retrieve_orbit_file(abs_zip, orbit_dir=str(orbit_dir_path), concatenate=True)
        log.info("轨道文件已下载: %s", orbit_file)
        return orbit_file

    raise FileNotFoundError(
        f"未找到轨道文件（allow_download=false）。\n"
        f"请将 .EOF 文件放入: {orbit_dir_path.resolve()}"
    )


# ── 4. COMPASS 处理（关键修复：次景 scratch=product）────────────────────────
def process_all_bursts_compass(
    ref_zip: str,
    sec_zip: str,
    orbit_ref: str,
    orbit_sec: str,
    burst_ids_ref: List[str],
    burst_ids_sec: List[str],
    dem_file: str,
    product_dir_ref: str,
    product_dir_sec: str,
    work_dir: str,
) -> None:
    """对所有 burst 依次运行 COMPASS s1_cslc。

    【关键修复】次景的 scratch_folder 必须 = product_folder，
              确保 geo2rdr 输出的 .off 文件能被 resample 正确读取。
    """
    from compass import s1_cslc

    # 【目录对齐官方】使用固定目录名
    scratch_ref = str(Path(work_dir) / "scratch")
    Path(scratch_ref).mkdir(parents=True, exist_ok=True)
    Path(product_dir_ref).mkdir(parents=True, exist_ok=True)
    Path(product_dir_sec).mkdir(parents=True, exist_ok=True)

    # 使用共视交集（若长度不一致则取交集）
    common_ids = list(set(burst_ids_ref) & set(burst_ids_sec))
    if not common_ids:
        raise ValueError("参考景与次景无共视 Burst")
    if len(common_ids) < len(burst_ids_ref):
        log.warning("Burst 取交集: %d → %d", len(burst_ids_ref), len(common_ids))

    for i, burst_id in enumerate(common_ids):
        log.info("  [%d/%d] COMPASS burst: %s", i + 1, len(common_ids), burst_id)

        # 参考景：scratch 独立
        yaml_ref = str(Path(work_dir) / "s1_cslc.yaml")  # 【对齐官方】单 YAML 覆盖
        _write_compass_yaml(
            zip_path=ref_zip,
            orbit_file=orbit_ref,
            burst_id=burst_id,
            is_reference=True,
            dem_file=dem_file,
            product_dir=product_dir_ref,
            scratch_dir=scratch_ref,
            output_dir=str(Path(work_dir) / "output"),
            ref_product_dir=None,
            yaml_out=yaml_ref,
        )
        s1_cslc.run(yaml_ref, "radar")

        # 次景：【关键】scratch = product_sec，确保 .off 路径一致
        yaml_sec = str(Path(work_dir) / "s1_cslc.yaml")  # 【对齐官方】覆盖写入
        _write_compass_yaml(
            zip_path=sec_zip,
            orbit_file=orbit_sec,
            burst_id=burst_id,
            is_reference=False,
            dem_file=dem_file,
            product_dir=product_dir_sec,
            scratch_dir=product_dir_sec,  # ← ⚠️ 关键修复：次景 scratch = product
            output_dir=str(Path(work_dir) / "output_sec"),
            ref_product_dir=product_dir_ref,
            yaml_out=yaml_sec,
        )
        s1_cslc.run(yaml_sec, "radar")

    log.info("所有 burst COMPASS 处理完成。")


def _write_compass_yaml(
    zip_path: str,
    orbit_file: str,
    burst_id: str,
    is_reference: bool,
    dem_file: str,
    product_dir: str,
    scratch_dir: str,
    output_dir: str,
    ref_product_dir: Optional[str],
    yaml_out: str,
) -> None:
    """从模板生成 COMPASS YAML（官方逻辑：次景 scratch=product）。"""
    import s1reader
    from hyp3_autorift.vend.testGeogrid import getPol

    if not YAML_TEMPLATE.exists():
        raise FileNotFoundError(f"COMPASS 模板未找到: {YAML_TEMPLATE}")

    with open(YAML_TEMPLATE, "r") as f:
        lines = f.readlines()

    abs_zip   = str(Path(zip_path).resolve())
    abs_orbit = str(Path(orbit_file).resolve())
    abs_dem   = str(Path(dem_file).resolve())

    pol = getPol(abs_zip, abs_orbit)
    is_cross_pol = pol.lower() in ("hv", "vh")

    if not is_reference:
        if ref_product_dir is None:
            raise ValueError("次景处理时必须提供 ref_product_dir")
        ref_burst_dirs = sorted(glob.glob(str(Path(ref_product_dir) / burst_id / "*")))
        ref_file = ref_burst_dirs[0] if ref_burst_dirs else str(Path(ref_product_dir) / burst_id)
    else:
        ref_file = ""

    # 【核心修复】次景 scratch_folder 必须 = product_folder
    if is_reference:
        final_scratch = str(Path(scratch_dir).resolve())
        final_product = str(Path(product_dir).resolve())
    else:
        final_product = str(Path(product_dir).resolve())
        final_scratch = final_product  # ← ⚠️ 关键：确保 .off 文件路径一致

    replacements = {
        "s1_image":       abs_zip,
        "s1_orbit_file":  abs_orbit,
        "burst_ids":      f"[{burst_id}]",
        "bool_reference": "True" if is_reference else "False",
        "s1_ref_file":    ref_file,
        "product_folder": final_product,
        "scratch_folder": final_scratch,
        "output_folder":  str(Path(output_dir).resolve()),
    }

    Path(yaml_out).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w") as out:
        for line in lines:
            new_line = line
            for placeholder, value in replacements.items():
                if placeholder in new_line:
                    new_line = new_line.replace(placeholder, value)
            if "'dem.tif'" in new_line or '"dem.tif"' in new_line:
                new_line = new_line.replace("dem.tif", abs_dem)
            if is_cross_pol and "co-pol" in new_line:
                new_line = new_line.replace("co-pol", "cross-pol")
            out.write(new_line)


# ── 5. 合并 burst 幅度图（官方重叠区逻辑 + 64px 缓冲）──────────────────────
def merge_burst_amplitudes(
    burst_ids: List[str],
    ref_zip: str,
    orbit_ref: str,
    product_dir_ref: str,
    product_dir_sec: str,
    amp_ref_path: str,
    amp_sec_path: str,
) -> Tuple[int, int]:
    """将所有 burst 的 CSLC 幅度图合并为单幅全景 GeoTIFF。

    【官方对齐】两层合并：
      1. 单 Swath 内 burst → 重叠区中点切分（消除 seam）
      2. 多 Swath → 非末端 Swath 添加 64px 无效缓冲（消除重采样伪影）
    """
    import s1reader
    from hyp3_autorift.vend.testGeogrid import getPol

    abs_zip   = str(Path(ref_zip).resolve())
    abs_orbit = str(Path(orbit_ref).resolve())
    pol = getPol(abs_zip, abs_orbit)

    # 按 Swath 分组
    bursts_by_swath: Dict[int, List[str]] = {}
    for burst_id in burst_ids:
        swath_num = int(burst_id.split("_")[-1][2:])  # "iw2" → 2
        bursts_by_swath.setdefault(swath_num, []).append(burst_id)

    swaths_sorted = sorted(bursts_by_swath.keys())
    log.info("按 Swath 分组: %s", {s: len(bids) for s, bids in bursts_by_swath.items()})

    # Step 1: 合并单 Swath 内 burst
    temp_ref_files = []
    temp_sec_files = []
    for swath in swaths_sorted:
        burst_ids_swath = bursts_by_swath[swath]
        log.info("合并 Swath %d: %d burst(s)", swath, len(burst_ids_swath))
        
        ref_temp = str(Path(amp_ref_path).parent / f"ref_swath_iw{swath}.tif")
        sec_temp = str(Path(amp_sec_path).parent / f"sec_swath_iw{swath}.tif")
        
        _merge_bursts_in_swath(
            burst_ids=burst_ids_swath,
            ref_zip=ref_zip,
            orbit_ref=orbit_ref,
            pol=pol,
            product_dir_ref=product_dir_ref,
            product_dir_sec=product_dir_sec,
            out_ref_path=ref_temp,
            out_sec_path=sec_temp,
        )
        temp_ref_files.append(ref_temp)
        temp_sec_files.append(sec_temp)

    # Step 2: 合并多 Swath（官方 64px 缓冲策略）
    log.info("合并 %d 个 Swath → 最终幅度图", len(swaths_sorted))
    num_az, num_rng = _merge_swaths(
        temp_ref_files=temp_ref_files,
        temp_sec_files=temp_sec_files,
        swaths=swaths_sorted,
        ref_zip=ref_zip,
        orbit_ref=orbit_ref,
        pol=pol,
        out_ref_path=amp_ref_path,
        out_sec_path=amp_sec_path,
    )

    # 清理临时文件
    for f in temp_ref_files + temp_sec_files:
        Path(f).unlink(missing_ok=True)
    log.info("临时文件已清理")

    return num_az, num_rng


def _merge_bursts_in_swath(
    burst_ids: List[str],
    ref_zip: str,
    orbit_ref: str,
    pol: str,
    product_dir_ref: str,
    product_dir_sec: str,
    out_ref_path: str,
    out_sec_path: str,
) -> Tuple[int, int]:
    """合并单个 Swath 内的多个 burst（官方重叠区中点切分）。"""
    import s1reader

    abs_zip = str(Path(ref_zip).resolve())
    abs_orbit = str(Path(orbit_ref).resolve())

    # 加载 burst 对象
    swath_num = int(burst_ids[0].split("_")[-1][2:])
    bursts = [
        b for b in s1reader.load_bursts(abs_zip, abs_orbit, swath_num, pol)
        if _burst_to_isce3_id(b) in burst_ids
    ]
    bursts = sorted(bursts, key=lambda b: b.sensing_start)

    if len(bursts) != len(burst_ids):
        raise RuntimeError(f"burst 数量不匹配: 期望 {len(burst_ids)}, 找到 {len(bursts)}")

    # 找到 SLC 文件路径
    ref_slc_paths = [_find_slc_tif(product_dir_ref, bid) for bid in burst_ids]
    sec_slc_paths = [_find_slc_tif(product_dir_sec, bid) for bid in burst_ids]

    num_bursts = len(bursts)

    # 单 burst 直接复制
    if num_bursts == 1:
        ref_arr = _read_amplitude(ref_slc_paths[0])
        sec_arr = _read_amplitude(sec_slc_paths[0])
        _write_slc_gdal(ref_arr, out_ref_path)
        _write_slc_gdal(sec_arr, out_sec_path)
        return ref_arr.shape

    # 计算方位向偏移（官方 get_azimuth_reference_offsets 逻辑）
    az_time_interval = bursts[0].azimuth_time_interval
    sensing_start_0 = bursts[0].sensing_start

    az_offsets = []
    for burst in bursts:
        az_offset = burst.sensing_start + timedelta(seconds=burst.first_valid_line * az_time_interval)
        start_idx = int(np.round((az_offset - sensing_start_0).total_seconds() / az_time_interval))
        end_idx = start_idx + (burst.last_valid_line - burst.first_valid_line) + 1
        az_offsets.append((start_idx, end_idx))

    # 计算总行数
    last_burst = bursts[-1]
    burst_length = timedelta(seconds=(last_burst.last_valid_line - last_burst.first_valid_line + 1) * az_time_interval)
    sensing_end = last_burst.sensing_start + timedelta(seconds=last_burst.first_valid_line * az_time_interval) + burst_length
    num_az_lines = 1 + int(np.round((sensing_end - sensing_start_0).total_seconds() / az_time_interval))
    
    # 2026/04/03/15/18
    #burst_length = timedelta(seconds=(burst.shape[0] - 1) * az_time_interval)
    # = last_burst.sensing_start + burst_length
    #num_az_lines = 1 + int(np.round((sensing_end - sensing_start_0).total_seconds() / az_time_interval))

    # 距离向尺寸
    first_ds = gdal.Open(ref_slc_paths[0], gdal.GA_ReadOnly)
    num_rng_samples = first_ds.RasterXSize
    first_ds = None

    log.info("Swath %d 合并尺寸: az=%d, rng=%d", swath_num, num_az_lines, num_rng_samples)

    ref_merged = np.zeros((num_az_lines, num_rng_samples), dtype=np.float32)
    sec_merged = np.zeros((num_az_lines, num_rng_samples), dtype=np.float32)

    # 逐个 burst 合并（重叠区中点切分）
    for idx, (burst, ref_path, sec_path, (burst_start, burst_end)) in enumerate(
        zip(bursts, ref_slc_paths, sec_slc_paths, az_offsets)
    ):
        ref_arr = _read_amplitude(ref_path)
        sec_arr = _read_amplitude(sec_path)
        rng_slice = slice(burst.first_valid_sample, burst.last_valid_sample)

        # 计算合并切片（官方逻辑）
        if idx == 0:
            next_start = az_offsets[idx + 1][0]
            overlap = (burst_end - next_start) // 2
            merge_start, merge_end = burst_start, burst_end - overlap
            slc_start, slc_end = burst.first_valid_line, burst.last_valid_line + 1 - overlap
        elif idx == num_bursts - 1:
            prev_end = az_offsets[idx - 1][1]
            overlap = (prev_end - burst_start) // 2
            merge_start, merge_end = burst_start + overlap, burst_end
            slc_start, slc_end = burst.first_valid_line + overlap, burst.last_valid_line + 1
        else:
            prev_end, next_start = az_offsets[idx - 1][1], az_offsets[idx + 1][0]
            ov1, ov2 = (prev_end - burst_start) // 2, (burst_end - next_start) // 2
            merge_start, merge_end = burst_start + ov1, burst_end - ov2
            slc_start, slc_end = burst.first_valid_line + ov1, burst.last_valid_line + 1 - ov2

        ref_merged[merge_start:merge_end, rng_slice] = ref_arr[slc_start:slc_end, rng_slice]
        sec_merged[merge_start:merge_end, rng_slice] = sec_arr[slc_start:slc_end, rng_slice]

    _write_slc_gdal(ref_merged, out_ref_path)
    _write_slc_gdal(sec_merged, out_sec_path)
    log.info("Swath %d 合并完成: %s", swath_num, out_ref_path)

    return num_az_lines, num_rng_samples
'''

def _merge_bursts_in_swath(
    burst_ids: List[str],
    ref_zip: str,
    orbit_ref: str,
    pol: str,
    product_dir_ref: str,
    product_dir_sec: str,
    out_ref_path: str,
    out_sec_path: str,
) -> Tuple[int, int]:
    """合并单个 Swath 内的多个 burst（官方重叠区中点切分）。"""
    import s1reader

    abs_zip = str(Path(ref_zip).resolve())
    abs_orbit = str(Path(orbit_ref).resolve())

    # 1. 加载 swath 和 burst 对象
    swath_num = int(burst_ids[0].split("_")[-1][2:])
    bursts = [
        b for b in s1reader.load_bursts(abs_zip, abs_orbit, swath_num, pol)
        if _burst_to_isce3_id(b) in burst_ids
    ]
    bursts = sorted(bursts, key=lambda b: b.sensing_start)

    if len(bursts) != len(burst_ids):
        raise RuntimeError(f"burst 数量不匹配: 期望 {len(burst_ids)}, 找到 {len(bursts)}")

    # 2. 找到对应的 SLC 文件路径
    ref_slc_paths = [_find_slc_tif(product_dir_ref, bid) for bid in burst_ids]
    sec_slc_paths = [_find_slc_tif(product_dir_sec, bid) for bid in burst_ids]

    num_bursts = len(bursts)

    # 单 burst 直接复制，无需合并
    if num_bursts == 1:
        ref_arr = _read_amplitude(ref_slc_paths[0])
        sec_arr = _read_amplitude(sec_slc_paths[0])
        _write_slc_gdal(ref_arr, out_ref_path)
        _write_slc_gdal(sec_arr, out_sec_path)
        return ref_arr.shape

    # 预读首个 burst 获取尺寸（COMPASS 输出为 full burst）
    ref_arr_0 = _read_amplitude(ref_slc_paths[0])
    sec_arr_0 = _read_amplitude(sec_slc_paths[0])
    full_az_lines = ref_arr_0.shape[0]      # COMPASS 输出的完整方位向行数
    num_rng_samples = ref_arr_0.shape[1]

    # 计算方位向参考偏移（对齐官方 get_azimuth_reference_offsets 逻辑）
    az_time_interval = bursts[0].azimuth_time_interval
    sensing_start_0 = bursts[0].sensing_start

    az_offsets = []
    for burst in bursts:
        az_offset_time = burst.sensing_start + timedelta(seconds=burst.first_valid_line * az_time_interval)
        start_idx = int(np.round((az_offset_time - sensing_start_0).total_seconds() / az_time_interval))
        end_idx = start_idx + (burst.last_valid_line - burst.first_valid_line) + 1
        az_offsets.append((start_idx, end_idx))

    # 计算全景方位向行数（基于最后一个 Burst 的结束时间）
    last_burst = bursts[-1]
    burst_length = timedelta(seconds=(full_az_lines - 1.0) * az_time_interval)
    sensing_end = last_burst.sensing_start + burst_length
    num_az_lines = 1 + int(np.round((sensing_end - sensing_start_0).total_seconds() / az_time_interval))

    log.info("Swath %d 合并尺寸: az=%d (预分配), rng=%d", swath_num, num_az_lines, num_rng_samples)

    ref_merged = np.zeros((num_az_lines, num_rng_samples), dtype=np.float32)
    sec_merged = np.zeros((num_az_lines, num_rng_samples), dtype=np.float32)

    # 逐个 burst 合并（重叠区中点切分）
    for idx, (burst, ref_path, sec_path, (burst_start, burst_end)) in enumerate(
        zip(bursts, ref_slc_paths, sec_slc_paths, az_offsets)
    ):
        # 按需读取 burst，避免重复 IO
        if idx == 0:
            ref_arr, sec_arr = ref_arr_0, sec_arr_0
        else:
            ref_arr = _read_amplitude(ref_path)
            sec_arr = _read_amplitude(sec_path)

        rng_slice = slice(burst.first_valid_sample, burst.last_valid_sample)

        # 计算重叠区 overlap 并确定切片
        if idx == 0:  # 第一个 burst
            next_start = az_offsets[idx + 1][0]
            overlap = (burst_end - next_start) // 2
            merge_start, merge_end = burst_start, burst_end - overlap
            slc_start, slc_end = burst.first_valid_line, burst.last_valid_line + 1 - overlap

        elif idx == num_bursts - 1:  # 最后一个 burst
            prev_end = az_offsets[idx - 1][1]
            overlap = (prev_end - burst_start) // 2
            merge_start, merge_end = burst_start + overlap, burst_end
            slc_start, slc_end = burst.first_valid_line + overlap, burst.last_valid_line + 1

        else:  # 中间 burst
            prev_end = az_offsets[idx - 1][1]
            next_start = az_offsets[idx + 1][0]
            ov_prev = (prev_end - burst_start) // 2
            ov_next = (burst_end - next_start) // 2
            merge_start, merge_end = burst_start + ov_prev, burst_end - ov_next
            slc_start, slc_end = burst.first_valid_line + ov_prev, burst.last_valid_line + 1 - ov_next

        # 写入合并结果
        ref_merged[merge_start:merge_end, rng_slice] = ref_arr[slc_start:slc_end, rng_slice]
        sec_merged[merge_start:merge_end, rng_slice] = sec_arr[slc_start:slc_end, rng_slice]

    _write_slc_gdal(ref_merged, out_ref_path)
    _write_slc_gdal(sec_merged, out_sec_path)
    log.info("Swath %d 合并完成: %s", swath_num, out_ref_path)

    return num_az_lines, num_rng_samples
'''



def _merge_swaths(
    temp_ref_files: List[str],
    temp_sec_files: List[str],
    swaths: List[int],
    ref_zip: str,
    orbit_ref: str,
    pol: str,
    out_ref_path: str,
    out_sec_path: str,
) -> Tuple[int, int]:
    """合并多个 Swath（官方 64px 缓冲策略）。"""
    import s1reader

    abs_zip = str(Path(ref_zip).resolve())
    abs_orbit = str(Path(orbit_ref).resolve())

    # 加载首个 Swath 参数
    bursts_first = s1reader.load_bursts(abs_zip, abs_orbit, swaths[0], pol)
    burst_0 = bursts_first[0]
    az_time_interval = burst_0.azimuth_time_interval
    range_pixel_spacing = burst_0.range_pixel_spacing
    first_starting_range = burst_0.starting_range

    # 收集 Swath 信息
    swath_info = []
    global_sensing_start = global_sensing_end = None

    for swath, ref_temp, sec_temp in zip(swaths, temp_ref_files, temp_sec_files):
        bursts = s1reader.load_bursts(abs_zip, abs_orbit, swath, pol)
        burst = bursts[0]

        if global_sensing_start is None or burst.sensing_start < global_sensing_start:
            global_sensing_start = burst.sensing_start

        ds = gdal.Open(ref_temp, gdal.GA_ReadOnly)
        num_lines = ds.RasterYSize
        ds = None
        swath_sensing_end = burst.sensing_start + timedelta(seconds=(num_lines - 1) * az_time_interval)
        if global_sensing_end is None or swath_sensing_end > global_sensing_end:
            global_sensing_end = swath_sensing_end

        rng_offset = int(np.floor((burst.starting_range - first_starting_range) / range_pixel_spacing))
        ref_arr = _read_amplitude(ref_temp)
        sec_arr = _read_amplitude(sec_temp)

        swath_info.append({
            "swath": swath,
            "rng_offset": rng_offset,
            "num_az": ref_arr.shape[0],
            "num_rng": ref_arr.shape[1],
            "ref_arr": ref_arr,
            "sec_arr": sec_arr,
            "sensing_start": burst.sensing_start,
        })

    # 计算全局尺寸
    sensing_duration = (global_sensing_end - global_sensing_start).total_seconds()
    total_az_lines = 1 + int(np.round(sensing_duration / az_time_interval))
    max_rng_end = max(info["rng_offset"] + info["num_rng"] for info in swath_info)
    total_rng_samples = max_rng_end

    log.info("全景合并尺寸: az=%d, rng=%d", total_az_lines, total_rng_samples)

    ref_final = np.zeros((total_az_lines, total_rng_samples), dtype=np.float32)
    sec_final = np.zeros((total_az_lines, total_rng_samples), dtype=np.float32)

    # 逐个 Swath 合并（官方 64px 缓冲）
    for info in swath_info:
        az_offset = int(np.round(
            (info["sensing_start"] - global_sensing_start).total_seconds() / az_time_interval
        ))
        az_slice = slice(az_offset, az_offset + info["num_az"])
        rng_slice = slice(info["rng_offset"], info["rng_offset"] + info["num_rng"])

        # 【关键】非最后一个 Swath 剔除边缘 64px 重采样伪影
        invalid_buf = 64 if info["swath"] != swaths[-1] else 0
        ref_mask = (ref_final[az_slice, rng_slice] == 0) & (info["ref_arr"] != 0)
        sec_mask = (sec_final[az_slice, rng_slice] == 0) & (info["sec_arr"] != 0)

        if invalid_buf > 0:
            ref_mask[:, -invalid_buf:] = False
            sec_mask[:, -invalid_buf:] = False

        ref_final[az_slice, rng_slice][ref_mask] = info["ref_arr"][ref_mask]
        sec_final[az_slice, rng_slice][sec_mask] = info["sec_arr"][sec_mask]

    _write_slc_gdal(ref_final, out_ref_path)
    _write_slc_gdal(sec_final, out_sec_path)
    log.info("全景合并完成: %s, %s", out_ref_path, out_sec_path)

    return total_az_lines, total_rng_samples


def _find_slc_tif(product_dir: str, burst_id: str) -> str:
    """在 COMPASS 产品目录中查找 burst 的 .slc.tif 文件。"""
    patterns = [
        str(Path(product_dir) / burst_id / "**" / "*.slc.tif"),
        str(Path(product_dir) / burst_id / "*" / "*.slc.tif"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"未找到 CSLC 输出。product_dir={product_dir}, burst_id={burst_id}\n"
        f"请确认 COMPASS 正常完成。"
    )


def _read_amplitude(slc_tif: str) -> np.ndarray:
    """读取复数 SLC，返回幅度 float32 数组。"""
    ds = gdal.Open(slc_tif, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return np.abs(arr).astype(np.float32)


def _write_slc_gdal(arr: np.ndarray, path: str) -> None:
    """将幅度图写为 GeoTIFF（无地理参考）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


# ── 6. 加载全景元数据（保持原接口）─────────────────────────────────────────
def load_slc_metadata_pair(
    ref_zip: str,
    sec_zip: str,
    orbit_ref: str,
    orbit_sec: str,
    swaths: List[int],
):
    """使用 loadMetadataSlc 加载全景元数据对。"""
    from hyp3_autorift.vend.testGeogrid import loadMetadataSlc

    abs_ref   = str(Path(ref_zip).resolve())
    abs_sec   = str(Path(sec_zip).resolve())
    abs_orb_r = str(Path(orbit_ref).resolve())
    abs_orb_s = str(Path(orbit_sec).resolve())

    log.info("加载参考景 SLC 元数据（Swath %s）...", swaths)
    meta_r = loadMetadataSlc(abs_ref, abs_orb_r, swaths=swaths)

    log.info("加载次景 SLC 元数据（仅读取时间）...")
    meta_temp = loadMetadataSlc(abs_sec, abs_orb_s, swaths=swaths)

    meta_s = copy.copy(meta_r)
    meta_s.sensingStart = meta_temp.sensingStart
    meta_s.sensingStop  = meta_temp.sensingStop

    log.info(
        "参考景: %s → %s",
        str(meta_r.sensingStart)[:19],
        str(meta_r.sensingStop)[:19],
    )
    log.info(
        "次景:   %s → %s",
        str(meta_s.sensingStart)[:19],
        str(meta_s.sensingStop)[:19],
    )
    return meta_r, meta_s
