"""
temporal.py
===========
Assign VelocityRecord objects to temporal groups for all supported modes.

Antarctic season definitions (opposite of Northern Hemisphere):
  Summer : Dec, Jan, Feb  (Dec → current year labelled yyyy-yyyy+1)
  Autumn : Mar, Apr, May
  Winter : Jun, Jul, Aug
  Spring : Sep, Oct, Nov
"""

from __future__ import annotations
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import TemporalMode
from data_sources import VelocityRecord

log = logging.getLogger(__name__)


def _group_key(dt: datetime, mode: TemporalMode, t0: datetime) -> str:
    m, y = dt.month, dt.year

    if mode == TemporalMode.MONTHLY:
        return dt.strftime("%Y-%m")

    elif mode == TemporalMode.SEASONAL:
        if m == 12:
            return f"{y}-{y+1}-Summer"
        elif m in (1, 2):
            return f"{y-1}-{y}-Summer"
        elif m in (3, 4, 5):
            return f"{y}-Autumn"
        elif m in (6, 7, 8):
            return f"{y}-Winter"
        else:
            return f"{y}-Spring"

    elif mode == TemporalMode.YEARLY:
        return str(y)

    elif mode == TemporalMode.ALL:
        return "all"

    else:
        # fixed_N  → N-day bins counted from earliest mid_date
        win_days = int(mode.value.replace("fixed_", ""))
        days_diff = (dt - t0).days
        bin_num   = days_diff // win_days
        bin_start = t0 + timedelta(days=bin_num * win_days)
        return bin_start.strftime("%Y-%m-%d")


def assign_groups(
    records: List[VelocityRecord],
    mode: TemporalMode,
) -> Dict[str, List[VelocityRecord]]:
    """
    Returns an ordered dict  {group_label: [VelocityRecord, ...]}.
    Groups are sorted chronologically (natural sort for seasonal labels too).
    """
    if not records:
        return {}

    t0 = min(r.mid_date for r in records)
    groups: Dict[str, List[VelocityRecord]] = defaultdict(list)

    for rec in records:
        key = _group_key(rec.mid_date, mode, t0)
        groups[key].append(rec)

    # Sort groups: for seasonal labels a simple sort is chronologically sensible
    sorted_groups = dict(sorted(groups.items()))

    log.info("Mode=%s  →  %d groups from %d records",
             mode.value, len(sorted_groups), len(records))
    for g, recs in sorted_groups.items():
        sensors = defaultdict(int)
        for r in recs:
            sensors[r.sensor] += 1
        log.debug("  %-22s  %3d records  %s", g, len(recs), dict(sensors))

    return sorted_groups


def write_group_log(
    output_path: str,
    mode: TemporalMode,
    groups: Dict[str, List[VelocityRecord]],
    cfg,
    group_stats: Optional[Dict[str, list]] = None,
) -> None:
    """Write a human-readable log of all group assignments.

    When group_stats is provided (populated after processing), each record
    shows its bbox coverage fraction and a participation marker (used / excluded).
    group_stats format: {group_name: [{"record", "coverage_frac", "participated",
                                       "skip_reason"}, ...]}
    """
    total         = sum(len(v) for v in groups.values())
    cov_threshold = getattr(cfg, "min_coverage_frac", 0.0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Temporal mode  : {mode.value}\n")
        f.write(f"ITS_LIVE dirs  : {', '.join(getattr(cfg, 'itslive_dirs', [getattr(cfg, 'itslive_dir', '')]))}\n")
        f.write(f"GAMMA dirs     : {', '.join(getattr(cfg, 'gamma_dirs',   [getattr(cfg, 'gamma_dir',   '')]))}\n")
        f.write(f"BBox (latlon)  : {cfg.bbox_latlon}\n")
        f.write(f"BBox (3031)    : {cfg.bbox_3031}\n")
        f.write(f"Cov. threshold : {cov_threshold*100:.1f}%\n")
        f.write(f"Total records  : {total}\n")
        f.write(f"Total groups   : {len(groups)}\n")
        f.write(f"Generated      : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 80 + "\n\n")

        for i, (gname, recs) in enumerate(groups.items(), 1):
            stats_list = (group_stats or {}).get(gname)
            n_used = sum(1 for s in stats_list if s["participated"]) \
                     if stats_list else len(recs)
            n_excl = len(recs) - n_used
            suffix = (f"  [used {n_used}/{len(recs)}"
                      + (f", excluded {n_excl}" if n_excl else "")
                      + "]") if stats_list else ""

            f.write(f"[{i:3d}/{len(groups)}]  {gname}  ({len(recs)} records){suffix}\n")

            stat_by_label: Dict[str, dict] = {}
            if stats_list:
                for s in stats_list:
                    stat_by_label[s["record"].label] = s
                f.write(f"  {'#':>3}  {'source':7s}  {'sensor':10s}  "
                        f"{'date pair':17s}  {'mid date':10s}  "
                        f"{'span':>5s}  {'err':3s}  {'bbox_cov':>8s}  status\n")
                f.write("  " + "-" * 76 + "\n")

            for j, r in enumerate(recs, 1):
                stat = stat_by_label.get(r.label)
                if stat:
                    cov_pct = stat["coverage_frac"] * 100
                    status  = ("used" if stat["participated"]
                               else f"EXCLUDED  ({stat.get('skip_reason') or ''})")
                    marker  = "v" if stat["participated"] else "x"
                    f.write(
                        f"  {j:3d}.  {r.source:7s}  {r.sensor:10s}  "
                        f"{r.date1:%Y%m%d}-{r.date2:%Y%m%d}  "
                        f"mid={r.mid_date:%Y-%m-%d}  "
                        f"span={r.time_span_days:3d}d  "
                        f"err={'yes' if r.v_error_path else 'no ':3s}  "
                        f"cov={cov_pct:5.1f}%  [{marker}] {status}\n"
                    )
                else:
                    f.write(
                        f"  {j:3d}.  {r.source:7s}  {r.sensor:10s}  "
                        f"{r.date1:%Y%m%d}-{r.date2:%Y%m%d}  "
                        f"mid={r.mid_date:%Y-%m-%d}  "
                        f"span={r.time_span_days:3d}d  "
                        f"err={'yes' if r.v_error_path else 'no'}\n"
                    )
            f.write("\n")
