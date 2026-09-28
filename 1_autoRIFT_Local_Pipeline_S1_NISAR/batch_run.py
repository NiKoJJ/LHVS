#!/usr/bin/env python3
"""
batch_run.py — 批量运行 autoRIFT SAR Local Pipeline（S1 & NISAR）
功能：
  1. 自动扫描数据目录，按时间间隔配对（支持 S1 .zip 和 NISAR .h5）
  2. 为每对生成永久 config.yaml 并存放于 {output_dir}/{pair_name}/
  3. 实时流式输出子进程日志（解决终端"卡住"问题）
  4. 支持断点续算、并行处理、超时控制
  5. 完整兼容现有 run_pipeline.py 与 config.yaml / config_nisar.yaml 结构

用法（S1，与旧版完全兼容）：
  python batch_run.py --data-dir /data1/SLC --orbit-dir /data2/Orbits \
    --output-dir ./results --base-config config.yaml --interval 12 --dry-run

用法（NISAR，--orbit-dir 无需提供）：
  python batch_run.py --data-dir /home/junjie/Downloads/RSLC/part2 \
    --output-dir ./results_part2 --base-config config_nisar.yaml \
    --sensor nisar --interval 12 --dry-run

用法（自动检测传感器）：
  python batch_run.py --data-dir /path/to/data \
    --output-dir ./results --base-config config.yaml --interval 12
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 终端编码兼容 ─────────────────────────────────────────────────────
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  文件名正则
# ══════════════════════════════════════════════════════════════════════════════

# Sentinel-1 SLC .zip（兼容 S1A/S1B/S1C）
_SLC_RE = re.compile(
    r'^(S1[ABC]_IW_SLC__1S[A-Z]{2}_(\d{8})T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4})\.zip$',
    re.IGNORECASE,
)

# NISAR RSLC/GSLC .h5
# 示例：NISAR_L1_PR_RSLC_006_153_A_144_7700_SHNA_A_20251108T202545_20251108T202558_X05010_N_P_J_001.h5
_NISAR_RE = re.compile(
    r'^(NISAR_L\d_PR_[RG]SLC'           # 产品标识
    r'_\d{3}_\d{3}_[AB]'                # orbit / track / pass
    r'_\d{3}_\d{4}_\w{4}_[AB]'         # frame / subframe / pol / mode
    r'_(\d{8})T\d{6}'                   # 开始时间（date 捕获组）
    r'_\d{8}T\d{6}'                     # 结束时间
    r'_\w+_\w_\w_\w_\d{3})\.h5$',      # 版本 / 质量 / 产品 / release
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
#  文件扫描
# ══════════════════════════════════════════════════════════════════════════════

def parse_s1_dir(data_dir: Path) -> List[Dict]:
    """扫描目录，解析 S1 SLC .zip 文件。"""
    records = []
    for f in sorted(data_dir.glob("*.zip")):
        m = _SLC_RE.match(f.name)
        if not m:
            log.warning("跳过非标准 S1 文件名: %s", f.name)
            continue
        records.append({
            "granule":   m.group(1),
            "date":      datetime.strptime(m.group(2), "%Y%m%d"),
            "satellite": m.group(1)[:3],   # S1A / S1B / S1C
            "sensor":    "S1",
            "path":      f.resolve(),
        })
    records.sort(key=lambda r: (r["date"], r["granule"]))
    log.info("找到 %d 个有效 S1 SLC .zip 文件", len(records))
    return records


def parse_nisar_dir(data_dir: Path) -> List[Dict]:
    """扫描目录，解析 NISAR RSLC/GSLC .h5 文件。"""
    records = []
    for f in sorted(data_dir.glob("*.h5")):
        m = _NISAR_RE.match(f.name)
        if not m:
            log.warning("跳过非标准 NISAR 文件名: %s", f.name)
            continue
        # HDF5 files can be left truncated by an interrupted download/copy.
        # Discovering those files here avoids creating a pair that is doomed to
        # fail much later inside ISCE3, and lets the remaining valid pairs run.
        try:
            import h5py
            with h5py.File(f, "r", libver="latest", swmr=True):
                pass
        except Exception as exc:
            log.warning(
                "跳过无法读取的 NISAR HDF5 文件: %s (%s). "
                "请重新复制或下载该文件。",
                f.name, exc,
            )
            continue
        records.append({
            "granule":   m.group(1),
            "date":      datetime.strptime(m.group(2), "%Y%m%d"),
            "satellite": "NISAR",
            "sensor":    "NISAR",
            "path":      f.resolve(),
        })
    records.sort(key=lambda r: (r["date"], r["granule"]))
    log.info("找到 %d 个有效 NISAR .h5 文件", len(records))
    return records


def detect_sensor_from_dir(data_dir: Path) -> str:
    """根据目录内容自动判断传感器类型。"""
    has_zip = any(data_dir.glob("S1[ABC]_IW_SLC__*.zip"))
    has_h5  = any(data_dir.glob("NISAR_*.h5"))
    if has_h5 and not has_zip:
        return "nisar"
    if has_zip and not has_h5:
        return "s1"
    if has_h5 and has_zip:
        log.warning("目录同时包含 S1 .zip 和 NISAR .h5，将优先使用 NISAR（可用 --sensor 覆盖）")
        return "nisar"
    return "s1"   # 默认回退


# ══════════════════════════════════════════════════════════════════════════════
#  配对逻辑（传感器无关）
# ══════════════════════════════════════════════════════════════════════════════

def build_pairs(
    records: List[Dict],
    interval_days: int,
    tolerance_days: int = 2,
    date_start: Optional[datetime] = None,
    date_end:   Optional[datetime] = None,
) -> List[Tuple[Dict, Dict]]:
    """按时间间隔 + 容差配对，同天优先选 S1A，避免重叠链。"""
    if date_start:
        records = [r for r in records if r["date"] >= date_start]
    if date_end:
        records = [r for r in records if r["date"] <= date_end]

    # 按日期分组，同日优先选 S1A（对 NISAR 无影响）
    by_date: Dict[datetime, List[Dict]] = {}
    for r in records:
        by_date.setdefault(r["date"], []).append(r)

    unique = []
    for d in sorted(by_date):
        s1a = [r for r in by_date[d] if r.get("satellite") == "S1A"]
        unique.append(s1a[0] if s1a else by_date[d][0])

    pairs, seen = [], set()
    for i, ref in enumerate(unique):
        target = ref["date"] + timedelta(days=interval_days)
        for sec in unique[i + 1:]:
            diff = abs((sec["date"] - target).days)
            if diff <= tolerance_days:
                key = (ref["date"], sec["date"])
                if key not in seen:
                    pairs.append((ref, sec))
                    seen.add(key)
                break
            if (sec["date"] - target).days > tolerance_days:
                break

    log.info("生成 %d 个配对（间隔 %d±%d 天）", len(pairs), interval_days, tolerance_days)
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  Config 生成（传感器感知）
# ══════════════════════════════════════════════════════════════════════════════

def _generate_config(
    ref:         Dict,
    sec:         Dict,
    orbit_dir:   Optional[str],
    base_config: str,
    output_dir:  str,
    pair_name:   str,
    sensor:      str,
) -> str:
    """为当前配对生成永久 config.yaml，传感器差异在此处理。

    关键：run_pipeline.py 用 base_results_dir / pair_name 作为工作目录，
    因此这里设置 base_results_dir = output_dir（批量输出根目录），
    pair_name 由此函数注入，从而保证每对独立隔离。
    """
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 传感器专属输入字段 ─────────────────────────────────────────────────
    if sensor.upper() == "S1":
        cfg.setdefault("input", {})
        cfg["input"]["reference_zip"] = str(ref["path"])
        cfg["input"]["secondary_zip"] = str(sec["path"])
        # S1 需要轨道目录
        if orbit_dir:
            cfg.setdefault("orbit", {})
            cfg["orbit"]["orbit_dir"] = str(orbit_dir)
    else:  # NISAR
        cfg.setdefault("input", {})
        cfg["input"]["reference_h5"] = str(ref["path"])
        cfg["input"]["secondary_h5"] = str(sec["path"])
        # NISAR 不使用 orbit 字段；若基础模板里有多余的 S1 字段也不影响

    # ── 输出路由：让 run_pipeline.py 把结果放到 output_dir/pair_name/ ──────
    # run_pipeline.py: work_dir = base_results_dir / pair_name
    cfg.setdefault("processing", {})
    cfg["processing"]["base_results_dir"] = str(output_dir)
    cfg["processing"]["pair_name"]        = pair_name

    # ── 写入 pair 专属目录 ────────────────────────────────────────────────
    pair_dir = Path(output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    config_path = pair_dir / f"{pair_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    log.debug("配置文件已写入: %s", config_path)
    return str(config_path)


# ══════════════════════════════════════════════════════════════════════════════
#  完成检测 & 单对运行
# ══════════════════════════════════════════════════════════════════════════════

def _is_completed(output_dir: str, pair_name: str) -> bool:
    """检查配对是否已完成（output_dir/pair_name/ 下存在 velocity_*.tif）。"""
    out_path = Path(output_dir) / pair_name
    return any(out_path.glob("velocity_*.tif"))


def _run_pair(
    ref:         Dict,
    sec:         Dict,
    args:        argparse.Namespace,
    base_config: str,
    sensor:      str,
) -> Tuple[str, bool, Optional[str]]:
    """执行单个配对处理（实时流式输出）。"""
    pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"

    # 断点续算
    if _is_completed(args.output_dir, pair_name):
        return pair_name, True, None

    try:
        config_path = _generate_config(
            ref=ref,
            sec=sec,
            orbit_dir=getattr(args, "orbit_dir", None),
            base_config=base_config,
            output_dir=args.output_dir,
            pair_name=pair_name,
            sensor=sensor,
        )

        cmd = [
            sys.executable,
            str(Path(args.pipeline_dir) / "run_pipeline.py"),
            "--config", config_path,
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.keep_intermediates:
            cmd.append("--keep-intermediates")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"]  = "1"
        env["PYTHONIOENCODING"]  = "utf-8"
        # 抑制 Numba/ISCE3 debug 洪流
        env["NUMBA_DEBUG"]       = "0"
        env["NUMBA_DISABLE_JIT"] = "0"
        env["NUMBA_NUM_THREADS"] = str(getattr(args, "n_threads", 4))

        log.info("[%s] 开始处理（传感器=%s）...", pair_name, sensor.upper())

        with subprocess.Popen(
            cmd,
            cwd=str(args.pipeline_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,   # 防止 fd 继承挂起
        ) as proc:
            for line in proc.stdout:
                print(f"[{pair_name}] {line.rstrip()}", flush=True)
            proc.wait()

        if proc.returncode != 0:
            return pair_name, False, f"退出码 {proc.returncode}"

        log.info("[%s] ✓ 成功", pair_name)
        return pair_name, True, None

    except subprocess.TimeoutExpired:
        return pair_name, False, f"超时 ({args.timeout_minutes} 分钟)"
    except Exception as e:
        return pair_name, False, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI & 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="批量运行 autoRIFT SAR Local Pipeline（S1 & NISAR）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 路径参数 ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--data-dir", "--slc-dir",   # --slc-dir 向后兼容
        dest="data_dir", required=True,
        help="数据文件目录（S1 .zip 或 NISAR .h5）",
    )
    p.add_argument(
        "--orbit-dir",
        default=None,
        help="S1 轨道文件目录 (.EOF)；NISAR 模式下可省略",
    )
    p.add_argument("--output-dir",   required=True, help="结果输出根目录（如 ./results）")
    p.add_argument("--pipeline-dir", default=".",   help="pipeline 根目录（含 run_pipeline.py）")
    p.add_argument("--base-config",  default="config.yaml", help="基础配置文件模板路径")

    # ── 传感器选择 ────────────────────────────────────────────────────────────
    p.add_argument(
        "--sensor",
        choices=["s1", "nisar", "auto"],
        default="auto",
        help="传感器类型（auto=自动检测目录内容）",
    )

    # ── 配对参数 ──────────────────────────────────────────────────────────────
    p.add_argument("--interval",  type=int, default=12, help="目标时间间隔（天）")
    p.add_argument("--tolerance", type=int, default=2,  help="时间容差（±天）")
    p.add_argument("--date-start", help="起始日期 YYYY-MM-DD")
    p.add_argument("--date-end",   help="结束日期 YYYY-MM-DD")

    # ── 执行参数 ──────────────────────────────────────────────────────────────
    p.add_argument("--dry-run",          action="store_true",      help="仅打印配对，不执行")
    p.add_argument("--max-workers",      type=int, default=1,      help="并行进程数（1=串行）")
    p.add_argument("--timeout-minutes",  type=int, default=180,    help="单对处理超时（分钟）")
    p.add_argument("--log-file",         help="日志文件路径")
    p.add_argument("--verbose", "-v",    action="store_true",      help="输出 DEBUG 日志")
    p.add_argument(
        "--keep-intermediates",
        action="store_true", default=False,
        help="保留所有中间文件（透传给 run_pipeline.py；默认：仅保留 velocity_*.tif / *.nc）",
    )

    args = p.parse_args()

    # ── 初始化日志 ────────────────────────────────────────────────────────────
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
    data_dir     = Path(args.data_dir)
    output_dir   = Path(args.output_dir)
    pipeline_dir = Path(args.pipeline_dir)
    base_config  = Path(args.base_config)

    for path, name in [
        (data_dir,                   "--data-dir"),
        (base_config,                "--base-config"),
        (pipeline_dir/"run_pipeline.py", "--pipeline-dir"),
    ]:
        if not path.exists():
            log.error("%s 不存在: %s", name, path)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 传感器检测 ────────────────────────────────────────────────────────────
    sensor = args.sensor
    if sensor == "auto":
        sensor = detect_sensor_from_dir(data_dir)
        log.info("自动检测传感器类型: %s", sensor.upper())
    else:
        log.info("传感器类型（用户指定）: %s", sensor.upper())

    # S1 模式下 orbit-dir 必须提供
    if sensor == "s1":
        if not args.orbit_dir:
            log.error("S1 模式需要 --orbit-dir（NISAR 模式可省略）")
            sys.exit(1)
        orbit_dir = Path(args.orbit_dir)
        if not orbit_dir.exists():
            log.error("--orbit-dir 不存在: %s", orbit_dir)
            sys.exit(1)

    # ── 扫描数据文件 ──────────────────────────────────────────────────────────
    if sensor == "s1":
        records = parse_s1_dir(data_dir)
    else:  # nisar
        records = parse_nisar_dir(data_dir)

    if not records:
        log.error("未找到有效 %s 文件（目录：%s）", sensor.upper(), data_dir)
        sys.exit(1)

    # ── 配对 ──────────────────────────────────────────────────────────────────
    date_start = datetime.strptime(args.date_start, "%Y-%m-%d") if args.date_start else None
    date_end   = datetime.strptime(args.date_end,   "%Y-%m-%d") if args.date_end   else None
    pairs = build_pairs(records, args.interval, args.tolerance, date_start, date_end)

    if not pairs:
        log.error("未生成任何配对（共 %d 个文件，间隔 %d±%d 天）",
                  len(records), args.interval, args.tolerance)
        sys.exit(1)

    # ── 配对预览 ──────────────────────────────────────────────────────────────
    print(f"\n传感器: {sensor.upper()}  |  共 {len(pairs)} 个待处理配对：")
    for i, (ref, sec) in enumerate(pairs, 1):
        span   = (sec["date"] - ref["date"]).days
        pname  = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
        status = "✓ 已完成" if _is_completed(str(output_dir), pname) else "○ 待处理"
        sat    = f"{ref.get('satellite','?')}/{sec.get('satellite','?')}"
        print(f"  {i:3d}. {ref['date']:%Y-%m-%d} → {sec['date']:%Y-%m-%d}"
              f" ({span}d)  {sat}  [{status}]")
    print()

    if args.dry_run:
        log.info("干运行模式，退出")
        return

    # ── 执行处理 ──────────────────────────────────────────────────────────────
    log.info("开始批量处理（传感器=%s，并行度=%d）", sensor.upper(), args.max_workers)
    ok, fail, skip = 0, 0, 0
    failed_pairs: List[Tuple[str, str]] = []

    if args.max_workers == 1:
        for ref, sec in pairs:
            pname = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
            if _is_completed(str(output_dir), pname):
                log.info("[%s] 已完成，跳过", pname)
                skip += 1
                continue
            pname, success, err = _run_pair(ref, sec, args, str(base_config), sensor)
            if success:
                ok += 1
            else:
                log.error("[%s] ✗ 失败: %s", pname, err)
                fail += 1
                failed_pairs.append((pname, err))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_run_pair, ref, sec, args, str(base_config), sensor): (ref, sec)
                for ref, sec in pairs
            }
            for fut in as_completed(futures):
                ref, sec = futures[fut]
                pname = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
                try:
                    pname, success, err = fut.result()
                    if success:
                        ok += 1
                    else:
                        log.error("[%s] ✗ 失败: %s", pname, err)
                        fail += 1
                        failed_pairs.append((pname, err))
                except Exception as e:
                    log.error("[%s] ✗ 异常: %s", pname, e)
                    fail += 1
                    failed_pairs.append((pname, str(e)))

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"批量处理完成：{ok} 成功 | {fail} 失败 | {skip} 已跳过")
    if failed_pairs:
        print("\n失败配对（可重试）:")
        for pname, err in failed_pairs:
            print(f"  - {pname}: {err[:100]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
