#!/usr/bin/env python
"""Generate and inspect DEM/SPS Geogrid inputs without processing an image pair.

This creates the pre-Geogrid parameter pack (dhdx/dhdy/cs*/sr*/vx/vy/ssm).
The sensor-dependent window_*.tif files still require NISAR/S1 metadata and a
subsequent Geogrid run in ``run_pipeline.py``.
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.geogrid_builder import build_geogrid_inputs, prepare_geogrid_dem
from src.utils import get_raster_info, setup_logging


SPS_KEYS = (
    "dhdx_file", "dhdy_file",
    "vx0_file", "vy0_file",
    "search_range_x_file", "search_range_y_file",
    "chip_size_min_x_file", "chip_size_min_y_file",
    "chip_size_max_x_file", "chip_size_max_y_file",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DEM/SPS autoRIFT Geogrid input rasters only."
    )
    parser.add_argument("--config", default="config_nisar.yaml", help="YAML configuration path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <working_dir>/parameter_pack)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}

    inp = cfg.get("input") or {}
    proc = cfg.get("processing") or {}
    dem_file = inp.get("dem_file")
    if not dem_file or not Path(dem_file).exists():
        raise FileNotFoundError("input.dem_file 不存在；该工具至少需要一个 DEM。")

    output_dir = Path(args.output or Path(proc.get("working_dir", "work")) / "parameter_pack")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resolution_m = proc.get("output_resolution_m")
    geogrid_dem = prepare_geogrid_dem(
        dem_file=dem_file,
        output_dir=str(output_dir),
        output_resolution_m=float(output_resolution_m) if output_resolution_m is not None else None,
        reuse_existing=bool(proc.get("reuse_geogrid_dem", True)),
    )
    sps_cfg = cfg.get("sps_params") or {}
    sps_files = {key: sps_cfg.get(key) or None for key in SPS_KEYS}
    files = build_geogrid_inputs(
        dem_file=geogrid_dem,
        ssm_file=inp.get("ssm_file") or None,
        output_dir=str(output_dir),
        chip_size_min_m=(
            float(proc["chip_size_min_m"])
            if proc.get("chip_size_min_m") is not None else None
        ),
        chip_size_max_m=(
            float(proc["chip_size_max_m"])
            if proc.get("chip_size_max_m") is not None else None
        ),
        search_range_m_yr=float(proc.get("search_range_m_yr", 3000.0)),
        sps_files=sps_files,
    )
    info = get_raster_info(geogrid_dem)
    logging.getLogger(__name__).info(
        "参数包完成：%s | DEM 格网 %d × %d | %.3f × %.3f m",
        output_dir.resolve(), info["rows"], info["cols"],
        abs(info["x_res"]), abs(info["y_res"]),
    )
    print("\n已生成 Geogrid 前参数文件：")
    for name, path in files.items():
        print(f"  {name:8s} {path}")
    print("\n下一步：用 run_pipeline.py 执行 Geogrid，生成 window_*.tif 后再运行 autoRIFT。")


if __name__ == "__main__":
    setup_logging()
    main()
