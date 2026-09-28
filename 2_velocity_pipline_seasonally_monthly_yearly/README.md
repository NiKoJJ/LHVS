# Glacier Velocity Pipeline (`pipline-v5`)

This project builds weighted-average glacier velocity products from ITS_LIVE and GAMMA offset-tracking rasters.

It supports:

- ITS_LIVE and GAMMA data ingestion
- Bounding-box clipping and reprojection to a common EPSG:3031 grid
- Optional single-scene spatial filters
- Optional temporal outlier filtering
- Inverse-variance weighted averaging
- GeoTIFF, CSV, and PNG outputs

## Main Files

- `run_pipeline.py`: command-line entry point
- `pipeline.py`: main orchestration logic
- `config.py`: central configuration dataclass and enums
- `data_sources.py`: ITS_LIVE / GAMMA file scanning and parsing
- `spatial.py`: clipping, reprojection, and target grid creation
- `error_and_outlier.py`: error estimation and outlier filtering
- `weighted_avg.py`: inverse-variance weighted averaging
- `io_utils.py`: GeoTIFF output writing
- `visualize.py`: quick-look PNG plots
- `example.py`: Python API example

## Dependencies

The code imports the following main packages:

- `numpy`
- `pandas`
- `rasterio`
- `shapely`
- `pyproj`
- `matplotlib`
- `scipy`

Install them in your Python environment before running the pipeline.

## Input Data Layout

### ITS_LIVE

The pipeline scans recursively for `.tif` files and groups them by filename stem.

Expected band suffixes:

- `_vx.tif`
- `_vy.tif`
- `_v.tif`
- `_v_error.tif`
- `_vx_error.tif` (optional)
- `_vy_error.tif` (optional)

At minimum, each scene must contain:

- `vx`
- `vy`
- `v`

### GAMMA

Expected filename style:

- `yyyyMMdd-yyyyMMdd-Vx.tif`
- `yyyyMMdd-yyyyMMdd-Vy.tif`
- `yyyyMMdd-yyyyMMdd-V.tif`

GAMMA scenes do not need error rasters. Errors can be estimated during processing.

## Running From Command Line

The simplest entry point is `run_pipeline.py`.

Example:

```bash
python run_pipeline.py \
  --itslive-dir /path/to/itslive \
  --gamma-dir /path/to/gamma \
  --output-dir /path/to/output \
  --modes monthly seasonal yearly \
  --bbox-latlon 158.0 -68.5 163.5 -66.5 \
  --resolution 120 \
  --use-gamma
```

You must provide one bounding box:

- `--bbox-latlon LON_MIN LAT_MIN LON_MAX LAT_MAX`
- `--bbox-3031 X_MIN Y_MIN X_MAX Y_MAX`

Common options:

- `--modes`: `monthly`, `seasonal`, `yearly`, `all`, `fixed_6`, `fixed_12`, `fixed_18`, `fixed_30`, `fixed_60`
- `--sensors`: limit ITS_LIVE to `S1`, `S2`, or `LC`
- `--no-gamma`: disable GAMMA input
- `--spatial-mad` / `--no-spatial-mad`
- `--temporal-mad` / `--no-temporal-mad`
- `--iqr-filter` / `--no-iqr-filter`
- `--min-coverage`: minimum scene coverage fraction in the target bbox
- `--date-start` / `--date-end`: filter by mid-date
- `--n-workers`: parallel processing across groups
- `--io-threads`: parallel file reading inside each group
- `--no-plots`: skip PNG overview figures
- `--no-neff`: skip `N_eff` rasters

## Running From Python

You can also call the pipeline directly from Python with `PipelineConfig`.

See `example.py` for a ready-to-edit script.

Minimal pattern:

```python
from config import PipelineConfig, TemporalMode, VxErrorMode
from pipeline import run_pipeline

cfg = PipelineConfig(
    itslive_dir="/path/to/itslive",
    gamma_dir="/path/to/gamma",
    output_dir="/path/to/output",
    bbox_latlon=(158.0, -68.5, 163.5, -66.5),
    temporal_modes=[TemporalMode.MONTHLY],
    target_resolution=120.0,
    use_gamma=True,
    vx_error_mode=VxErrorMode.ISOTROPIC,
    gamma_error_mode=VxErrorMode.LOCAL_STD,
)

summaries = run_pipeline(cfg)
print(f"Processed {len(summaries)} groups")
```

## Output Structure

The output root directory will contain files such as:

- `scene_inventory.csv`
- `grid_info.json`
- `master_summary.csv`

For each temporal mode, a subdirectory is created, for example:

- `monthly/`
- `seasonal/`
- `yearly/`

Inside each mode directory:

- `summary_<mode>.csv`
- `group_log_<mode>.txt`
- `timeseries_summary.png` if plots are enabled

Inside each group directory, the pipeline may write:

- `<group>_vx.tif`
- `<group>_vy.tif`
- `<group>_v.tif`
- `<group>_vx_err.tif`
- `<group>_vy_err.tif`
- `<group>_v_err.tif`
- `<group>_v_synth.tif`
- `<group>_v_synth_err.tif`
- `<group>_neff.tif`
- `<group>_overview.png`

## Processing Order

Within each temporal group, processing is performed in this order:

1. Optional spatial MAD filtering
2. Optional Universal Median Test filtering
3. Coverage filtering
4. Optional temporal MAD filtering
5. Optional IQR filtering
6. Inverse-variance weighted average
7. GeoTIFF / CSV / PNG export

## Notes

- The default target CRS is `EPSG:3031`.
- If both `bbox_latlon` and `bbox_3031` are set, the pipeline uses the projected bbox logic defined in the code path.
- `v_synth` is recomputed from weighted `vx` and `vy`, and `v_synth_err` is propagated from `vx_err` and `vy_err`.
- For large regions or many scenes, start with smaller `--n-workers` and `--io-threads` values, then scale up.
