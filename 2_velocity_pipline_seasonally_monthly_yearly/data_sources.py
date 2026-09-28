"""
data_sources.py
===============
Scan and parse file names for:
  1. ITS_LIVE  – multi-sensor TIFs  (S1, S2, Landsat 8/9)
  2. GAMMA POT – Sentinel-1 TIFs    (yyyyMMdd-yyyyMMdd-Vx.tif)

Returns a unified list of VelocityRecord objects.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VelocityRecord:
    """One image-pair velocity dataset (all three components + optional errors)."""
    source:  str
    sensor:  str
    date1:   datetime
    date2:   datetime

    vx_path: Path
    vy_path: Path
    v_path:  Path

    v_error_path:  Optional[Path] = None
    vx_error_path: Optional[Path] = None   # present in some ITS_LIVE products
    vy_error_path: Optional[Path] = None   # present in some ITS_LIVE products

    # derived
    mid_date:       datetime = field(init=False)
    time_span_days: int      = field(init=False)
    label:          str      = field(init=False)

    def __post_init__(self):
        self.mid_date       = self.date1 + (self.date2 - self.date1) / 2
        self.time_span_days = (self.date2 - self.date1).days
        self.label          = (
            f"{self.date1:%Y%m%d}-{self.date2:%Y%m%d}-{self.source}-{self.sensor}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ITS_LIVE scanner
# ─────────────────────────────────────────────────────────────────────────────

# Ordered so longer suffixes are tested before shorter ones to avoid
# substring matches (e.g. _vx_error.tif must be checked before _vx.tif).
_ITSLIVE_SUFFIXES: Dict[str, str] = {
    "vx_error": "_vx_error.tif",
    "vy_error": "_vy_error.tif",
    "v_error":  "_v_error.tif",
    "vx":       "_vx.tif",
    "vy":       "_vy.tif",
    "v":        "_v.tif",
}

_SENSOR_RE = {
    "S1":   re.compile(r'^S1[ABC]_'),
    "S2":   re.compile(r'^S2[ABC]_'),
    "LC08": re.compile(r'^LC08_'),
    "LC09": re.compile(r'^LC09_'),
}


def _detect_sensor(name: str) -> Optional[str]:
    for k, pat in _SENSOR_RE.items():
        if pat.match(name):
            return k[:2] if k.startswith('LC') else k[:2]
    return None


def _parse_itslive_dates(name: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Field indices (1-based, split on '_'):
        S2  : fields 3  & 11  → idx 2 & 10
        S1  : skip empty tokens, then fparts[4] & fparts[14]
        LC  : fields 4  & 12  → idx 3 & 11
    """
    def _p(tok: str) -> Optional[datetime]:
        try:
            return datetime.strptime(tok[:8], "%Y%m%d")
        except (ValueError, TypeError):
            return None

    parts = name.split("_")
    name_u = name.upper()
    try:
        if name_u.startswith("S2"):
            return _p(parts[2]), _p(parts[10])
        elif name_u.startswith("S1"):
            fp = [p for p in parts if p]
            return _p(fp[4]), _p(fp[14])
        elif name_u.startswith("LC08") or name_u.startswith("LC09"):
            return _p(parts[3]), _p(parts[11])
    except IndexError:
        pass
    return None, None


def _full_sensor(name: str) -> str:
    name_u = name.upper()
    if name_u.startswith("S1A"): return "S1A"
    if name_u.startswith("S1B"): return "S1B"
    if name_u.startswith("S1C"): return "S1C"
    if name_u.startswith("S2A"): return "S2A"
    if name_u.startswith("S2B"): return "S2B"
    if name_u.startswith("LC08"): return "LC08"
    if name_u.startswith("LC09"): return "LC09"
    return "UNKNOWN"


def scan_itslive(
    data_dir: str | Path,
    sensor_filter: Optional[List[str]] = None,
    max_days: Optional[int] = None,
) -> List[VelocityRecord]:
    """
    Scan an ITS_LIVE directory for complete (vx, vy, v, v_error) file groups.
    Groups are identified by their common filename stem (everything before _vx.tif).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        log.error("ITS_LIVE dir not found: %s", data_dir)
        return []

    # ── collect all TIFs, classify by suffix ─────────────────────────────────
    from collections import defaultdict
    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for p in sorted(data_dir.rglob("*.tif")):
        fname = p.name
        for key, suf in _ITSLIVE_SUFFIXES.items():
            if fname.lower().endswith(suf.lower()):
                stem = fname[: -len(suf)]
                groups[stem][key] = p
                break

    records: List[VelocityRecord] = []
    skipped = 0

    for stem, band_paths in groups.items():
        # Require at minimum vx, vy, v
        if not all(k in band_paths for k in ("vx", "vy", "v")):
            skipped += 1
            continue

        d1, d2 = _parse_itslive_dates(stem)
        if d1 is None or d2 is None:
            log.debug("Date parse failed for stem: %s", stem)
            skipped += 1
            continue

        if d1 > d2:
            d1, d2 = d2, d1

        # sensor filter
        sensor_full = _full_sensor(stem)
        sensor_2    = sensor_full[:2]           # 'S1', 'S2', 'LC'
        if sensor_filter and sensor_2 not in sensor_filter:
            continue

        # time-span filter
        span = (d2 - d1).days
        if max_days and span > max_days:
            continue

        rec = VelocityRecord(
            source         = "itslive",
            sensor         = sensor_full,
            date1          = d1,
            date2          = d2,
            vx_path        = band_paths["vx"],
            vy_path        = band_paths["vy"],
            v_path         = band_paths["v"],
            v_error_path   = band_paths.get("v_error"),
            vx_error_path  = band_paths.get("vx_error"),
            vy_error_path  = band_paths.get("vy_error"),
        )
        records.append(rec)

    records.sort(key=lambda r: r.mid_date)
    log.info("ITS_LIVE: found %d records (%d skipped) in %s",
             len(records), skipped, data_dir)
    if records:
        sensors = {}
        for r in records:
            sensors[r.sensor] = sensors.get(r.sensor, 0) + 1
        log.info("  Sensor breakdown: %s", sensors)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# GAMMA POT scanner
# ─────────────────────────────────────────────────────────────────────────────

_GAMMA_RE = re.compile(
    r'^(\d{8})-(\d{8})-(?:V|Vx|Vy|v|vx|vy)\.tif$', re.IGNORECASE
)
_GAMMA_VX_RE = re.compile(r'^(\d{8})-(\d{8})-[Vv][xX]\.tif$')


def scan_gamma(
    data_dir: str | Path,
    max_days: Optional[int] = None,
) -> List[VelocityRecord]:
    """
    Scan GAMMA POT directory.
    File naming: yyyyMMdd-yyyyMMdd-V.tif / -Vx.tif / -Vy.tif
    No error files – sigma will be estimated from local_std later.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        log.error("GAMMA dir not found: %s", data_dir)
        return []

    # Find all Vx files to anchor each pair
    vx_files = [
        p for p in sorted(data_dir.rglob("*.tif"))
        if _GAMMA_VX_RE.match(p.name)
    ]

    records: List[VelocityRecord] = []
    skipped = 0

    for vx_path in vx_files:
        m = _GAMMA_VX_RE.match(vx_path.name)
        if not m:
            skipped += 1
            continue

        d1_str, d2_str = m.group(1), m.group(2)
        try:
            d1 = datetime.strptime(d1_str, "%Y%m%d")
            d2 = datetime.strptime(d2_str, "%Y%m%d")
        except ValueError:
            skipped += 1
            continue

        if d1 > d2:
            d1, d2 = d2, d1

        span = (d2 - d1).days
        if max_days and span > max_days:
            continue

        prefix = f"{d1_str}-{d2_str}"
        parent = vx_path.parent

        # Look for companion Vy and V files (case-insensitive)
        vy_path = _find_gamma_file(parent, prefix, ("Vy", "vy", "VY"))
        v_path  = _find_gamma_file(parent, prefix, ("V",  "v",  "V"))

        # Exclude Vx / Vy from V search
        if v_path and _GAMMA_VX_RE.match(v_path.name):
            v_path = None

        if vy_path is None or v_path is None:
            log.debug("Incomplete GAMMA pair: %s (vy=%s, v=%s)", prefix, vy_path, v_path)
            skipped += 1
            continue

        rec = VelocityRecord(
            source       = "gamma",
            sensor       = "GAMMA_S1",
            date1        = d1,
            date2        = d2,
            vx_path      = vx_path,
            vy_path      = vy_path,
            v_path       = v_path,
            v_error_path = None,    # no error file in GAMMA output
        )
        records.append(rec)

    records.sort(key=lambda r: r.mid_date)
    log.info("GAMMA: found %d records (%d skipped) in %s",
             len(records), skipped, data_dir)
    return records


def _find_gamma_file(parent: Path, prefix: str, suffixes: tuple) -> Optional[Path]:
    """Try multiple case variants for GAMMA companion file."""
    for suf in suffixes:
        p = parent / f"{prefix}-{suf}.tif"
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_records(cfg) -> List[VelocityRecord]:
    """
    Load and merge records from ITS_LIVE and/or GAMMA according to config.
    Applies optional date range filter.
    """
    from datetime import datetime as dt

    records: List[VelocityRecord] = []

    if cfg.use_itslive:
        for itslive_dir in cfg.itslive_dirs:
            records += scan_itslive(
                itslive_dir,
                sensor_filter=cfg.itslive_sensors,
                max_days=cfg.itslive_max_days,
            )

    if cfg.use_gamma:
        for gamma_dir in cfg.gamma_dirs:
            records += scan_gamma(gamma_dir, max_days=cfg.gamma_max_days)

    # Date range filter
    if cfg.date_start:
        t0 = dt.strptime(cfg.date_start, "%Y-%m-%d")
        records = [r for r in records if r.mid_date >= t0]
    if cfg.date_end:
        t1 = dt.strptime(cfg.date_end, "%Y-%m-%d")
        records = [r for r in records if r.mid_date <= t1]

    records.sort(key=lambda r: r.mid_date)

    sources = {}
    for r in records:
        sources[r.source] = sources.get(r.source, 0) + 1
    log.info("Total records loaded: %d  %s", len(records), sources)

    return records


def records_to_dataframe(records: List[VelocityRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "source":         r.source,
            "sensor":         r.sensor,
            "date1":          r.date1,
            "date2":          r.date2,
            "mid_date":       r.mid_date,
            "span_days":      r.time_span_days,
            "vx_path":        str(r.vx_path),
            "vy_path":        str(r.vy_path),
            "v_path":         str(r.v_path),
            "v_error_path":   str(r.v_error_path)  if r.v_error_path  else "",
            "vx_error_path":  str(r.vx_error_path) if r.vx_error_path else "",
            "vy_error_path":  str(r.vy_error_path) if r.vy_error_path else "",
            "has_v_error":    r.v_error_path  is not None,
            "has_vxy_error":  r.vx_error_path is not None,
        })
    return pd.DataFrame(rows)
