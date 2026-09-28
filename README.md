# LHVS

English | [简体中文](README_zh-CN.md)

Local horizontal ice velocity mapping and temporal compositing from optical and SAR satellite imagery.

LHVS provides Python workflows for deriving glacier and ice-shelf velocity fields from locally stored satellite images and combining image-pair measurements into temporal composites. It brings together autoRIFT feature tracking, configurable geospatial processing, outlier filtering, and uncertainty-weighted averaging for studies of ice motion and its variability.

The project includes optical tracking with Landsat and Sentinel-2, SAR tracking with Sentinel-1 and NISAR, and a separate compositing module for compatible velocity rasters. Users can process individual image pairs, select pairs for batch processing, and produce monthly, seasonal, annual, or fixed-window velocity maps.

LHVS was developed for Cook Ice Shelf research in East Antarctica. Its workflows can be adapted to other glacier and ice-shelf study areas by configuring the imagery, DEM, matching parameters, and analysis domain. Temporal-compositing defaults use Antarctic Polar Stereographic coordinates (EPSG:3031) and Antarctic seasons.

## Features

- **Local image processing:** work with downloaded imagery and user-provided auxiliary data.
- **Optical and SAR workflows:** derive velocity fields from Landsat, Sentinel-2, Sentinel-1, and NISAR products.
- **Configurable tracking:** define DEM coverage, prior velocities, search ranges, matching-window sizes, and output settings.
- **Batch processing:** select image pairs using temporal baselines, dates, and sensor-specific criteria.
- **Quality control:** apply spatial and temporal outlier filters before compositing.
- **Temporal composites:** combine velocity measurements with inverse-variance weighting and export uncertainty estimates.
- **Geospatial outputs:** generate GeoTIFF velocity maps, optional tracking NetCDFs, CSV summaries, and quick-look figures.

## Workflow

| Module | Input | Processing | Output |
| --- | --- | --- | --- |
| Optical tracking | Landsat or Sentinel-2 band images and a DEM | Preprocessing, GeogridOptical, and autoRIFT matching | Image-pair velocity GeoTIFFs and optional NetCDFs |
| SAR tracking | Sentinel-1 SLC ZIP or NISAR RSLC/GSLC HDF5 products and a DEM | Coregistration or overlap preparation, amplitude extraction, Geogrid, and autoRIFT matching | Image-pair velocity GeoTIFFs and optional NetCDFs |
| Temporal compositing | Single-band velocity rasters in supported ITS_LIVE or GAMMA naming conventions | Common-grid processing, quality control, and uncertainty-weighted averaging | Temporal velocity composites, uncertainty maps, summaries, and figures |

The modules can be run independently. To composite local tracking results, first export their three-band GeoTIFFs into a supported single-band layout, as described in [Temporal velocity composites](#temporal-velocity-composites).

## Repository structure

```text
.
├── README.md
├── README_zh-CN.md
├── 1_autoRIFT_Local_Pipeline_Optical/
│   ├── run_pipeline.py                 # Process one optical image pair
│   ├── batch_run.py                    # Select and process multiple pairs
│   ├── config.yaml
│   └── src/
│       ├── optical_processor.py
│       ├── geogrid_builder_optical.py
│       ├── autorift_runner_optical.py
│       └── utils.py
├── 1_autoRIFT_Local_Pipeline_S1_NISAR/
│   ├── run_pipeline.py                 # Process one Sentinel-1 or NISAR pair
│   ├── batch_run.py
│   ├── generate_parameter_pack.py      # Prepare DEM/SPS inputs without tracking
│   ├── config.yaml                     # Sentinel-1 configuration
│   ├── config_nisar.yaml               # NISAR configuration
│   ├── NISAR_MODULE_README.md           # NISAR implementation notes
│   └── src/
│       ├── s1_processor.py
│       ├── nisar_processor.py
│       ├── geogrid_builder.py
│       ├── autorift_runner.py
│       ├── utils.py
│       └── schemas/s1_cslc_template.yaml
└── 2_velocity_pipline_seasonally_monthly_yearly/
    ├── run_pipeline.py                 # Temporal-compositing command line
    ├── example.py                      # Python configuration example
    ├── config.py
    ├── pipeline.py
    ├── data_sources.py
    ├── spatial.py
    ├── temporal.py
    ├── error_and_outlier.py
    ├── weighted_avg.py
    ├── io_utils.py
    └── visualize.py
```

Run commands from the directory of the module you want to use. The optical and SAR modules each contain a package named `src`; keep their execution environments and import paths separate.

## Installation

LHVS uses Python scientific and geospatial libraries together with external image-tracking and coregistration software. Prepare the dependencies for the modules you intend to run; temporal compositing can be used without installing the full tracking stack.

| Component | Required software |
| --- | --- |
| Optical tracking | Python, NumPy, PyYAML, GDAL Python bindings (`osgeo`), OpenCV (`cv2`), autoRIFT and Geogrid |
| Sentinel-1 tracking | Tracking dependencies plus `hyp3_autorift`, COMPASS, `s1reader`, and a compatible ISCE3 environment |
| NISAR tracking | Tracking dependencies plus `hyp3_autorift`, `h5py`, ISCE3, and the `nisar.products` / `nisar.workflows` Python modules |
| NetCDF export | `xarray` and `netCDF4` |
| Temporal compositing | NumPy, pandas, rasterio, SciPy, pyproj, Shapely, and Matplotlib |

A Python 3.10+ environment is a practical starting point. For tracking, choose a Python version supported by the required geospatial and SAR dependencies. Linux or WSL is recommended for working with the SAR software stack.

To create an environment for **temporal compositing**:

```bash
conda create -n lhvs -c conda-forge python=3.11 numpy pandas rasterio scipy pyproj shapely matplotlib
conda activate lhvs
```

For image tracking, follow the upstream installation instructions and use mutually compatible versions:

- [autoRIFT and Geogrid](https://github.com/nasa-jpl/autoRIFT)
- [HyP3 autoRIFT](https://github.com/ASFHyP3/hyp3-autorift)
- [COMPASS](https://github.com/opera-adt/COMPASS)
- [ISCE3](https://github.com/isce-framework/isce3)

The tracking modules use interfaces from these packages, including `hyp3_autorift.vend.testGeogrid`. Check API compatibility when selecting versions and retain the environment information for reproducible processing.

## Getting started

Choose the workflow that matches your data:

- **Optical band images:** start with [Optical image-pair tracking](#optical-image-pair-tracking).
- **Sentinel-1 or NISAR products:** start with [Sentinel-1 and NISAR tracking](#sentinel-1-and-nisar-tracking).
- **Existing velocity rasters:** start with [Temporal velocity composites](#temporal-velocity-composites).

For image tracking, provide local satellite products and a projected DEM covering the study area. A stable-surface mask and spatially varying prior-velocity or matching-parameter rasters can also be supplied. Sentinel-1 processing requires suitable orbit information. Download and prepare these datasets separately from the code.

The commands below use example paths. Replace them with your own data and output locations, and adjust the study-area bounds and processing parameters before running.

## Optical image-pair tracking

The main processing path recognizes Landsat 4/5/7/8/9 and Sentinel-2 filenames. Typical default bands are Landsat 7/8/9 B8, Landsat 4/5 B2, and Sentinel-2 B08. Other bands can be selected explicitly in batch processing.

### Configure and process one pair

From the repository root:

```bash
cd 1_autoRIFT_Local_Pipeline_Optical
```

Edit `config.yaml`. At minimum, replace `input.reference_path`, `input.secondary_path`, and `input.dem_file` with existing files. Check all other local paths, including `input.ssm_file` and every `sps_params` entry. Set unavailable optional files to `null`.

Important settings include:

- `processing.base_results_dir`: output root; each pair receives its own subdirectory.
- `processing.epsg`: target projection; `null` uses the DEM projection.
- `processing.chip_size_min_m` / `chip_size_max_m`: matching-window sizes in metres, not output pixel spacing.
- `processing.search_range_m_yr`: search-range fallback when spatial search parameters are unavailable.
- `processing.velocity_max_m_yr`: upper speed limit used to reject implausible results.
- `processing.output_netcdf` and `save_offsets`: optional product generation.
- `sps_params`: optional prior velocities, search ranges, chip sizes, and surface slopes.

Use a suitable projected DEM and a study-specific extent. Preserve recognizable scene filenames because acquisition dates are parsed from them.

```bash
python run_pipeline.py --config config.yaml --keep-intermediates
```

`--keep-intermediates` retains parameter rasters, offsets, and diagnostic files. Without this flag, the optical runner removes everything in the pair directory except `velocity_*.tif` and `velocity_*.nc` after successful processing. Use a dedicated output directory, not a directory containing unrelated files.

### Batch processing

The following example uses Landsat band B5, a maximum temporal baseline of 60 days, and a minimum spatial overlap of 40%. The paths are placeholders; replace them before running.

```bash
python batch_run.py --scene-dir /path/to/landsat --output-dir /path/to/optical_results --platform landsat --bands B5 --pipeline-dir . --base-config config.yaml --interval 60 --min-overlap 0.4 --dry-run
```

Inspect the pair list, then remove `--dry-run` to process it. For Sentinel-2, use `--platform s2 --bands B08` with the appropriate scene directory.

The optical `--interval` specifies a **maximum baseline**, not an exact repeat interval. The script selects eligible pairs within that limit, subject to `--min-days`, overlap, date, and sensor filters. This pair selection does not perform an SBAS displacement inversion.

For optical batch processing, set `processing.base_results_dir` and `--output-dir` to the same **absolute path**: configuration placement and completion checks use `--output-dir`, while the tracking runner uses `base_results_dir`. To retain diagnostics, run an individual generated pair configuration with `--keep-intermediates`; the optical batch CLI does not forward this flag.

Use Landsat or Sentinel-2 for the documented optical workflow. The additional `planetscope` and `custom` batch-discovery options require compatible downstream filename, date, and metadata handling.

## Sentinel-1 and NISAR tracking

Return to the repository root and enter the SAR directory:

```bash
cd 1_autoRIFT_Local_Pipeline_S1_NISAR
```

### Sentinel-1

Edit `config.yaml` to specify:

- `input.reference_zip` and `secondary_zip`: local Sentinel-1 SLC ZIP archives.
- `input.dem_file`: local DEM; optionally provide a stable-surface mask.
- `orbit.orbit_dir`: orbit-file directory; `orbit.allow_download` controls whether missing orbit files may be retrieved.
- `burst_selection`: selection settings; inspect burst availability before processing.
- `processing` and `sps_params`: output, matching, prior-velocity, and spatial-parameter settings.

```bash
python run_pipeline.py --config config.yaml --list-bursts
python run_pipeline.py --config config.yaml --keep-intermediates
```

A batch example is:

```bash
python batch_run.py --sensor s1 --data-dir /path/to/s1_slc --orbit-dir /path/to/orbits --output-dir /path/to/s1_results --pipeline-dir . --base-config config.yaml --interval 12 --tolerance 2 --dry-run
```

Remove `--dry-run` to process the selected pairs. Add `--keep-intermediates` if needed. Unlike the optical selector, SAR `--interval` is a **target baseline**, accepted within `--tolerance`; it does not enumerate every possible baseline.

### NISAR

Edit `config_nisar.yaml`, including `input.reference_h5`, `secondary_h5`, `dem_file`, `nisar.frequency`, and the output and optional spatial-parameter paths.

The code branches by product type:

- **RSLC:** ISCE3 coregistration, amplitude extraction, and radar-geometry metadata preparation.
- **GSLC:** overlap cropping, amplitude extraction, and geocoded metadata preparation.

```bash
python run_pipeline.py --config config_nisar.yaml --keep-intermediates
```

For batch pair inspection:

```bash
python batch_run.py --sensor nisar --data-dir /path/to/nisar_h5 --output-dir /path/to/nisar_results --pipeline-dir . --base-config config_nisar.yaml --interval 12 --tolerance 2 --dry-run
```

`processing.output_resolution_m` controls the working DEM spacing used by Geogrid in the SAR pipeline. It is distinct from matching-window size and does not by itself establish the effective spatial resolution of the velocity measurements.

To prepare only the DEM/SPS inputs:

```bash
python generate_parameter_pack.py --config config_nisar.yaml --output /path/to/parameter_pack
```

This utility does not generate sensor-dependent `window_*.tif` files or velocity products; those require the subsequent Geogrid and tracking steps.

As in optical processing, the SAR runner cleans intermediate files unless `--keep-intermediates` is supplied. The SAR cleanup retains velocity GeoTIFFs, velocity NetCDFs, and YAML configuration files, but removes other files and subdirectories in the pair directory.

### Tracking product definitions

Both tracking workflows write `velocity_<pair_name>.tif` as a Float32 GeoTIFF:

| Band | Description | Unit |
| --- | --- | --- |
| 1 | `Vx_m_yr`: velocity along the output projection's x axis | m yr⁻¹ |
| 2 | `Vy_m_yr`: velocity along the output projection's y axis | m yr⁻¹ |
| 3 | `V_m_yr`: speed, √(Vx² + Vy²) | m yr⁻¹ |

The tracking GeoTIFF NoData value is `-32767`. Projected x/y components are not automatically interchangeable with geographic east/north components.

Optional `offset_<pair_name>.tif` contains four bands: `DX_px`, `DY_px`, `InterpMask`, and `ChipSizeX`. Optional NetCDF files contain velocity fields, coordinates, projection information, matching diagnostics, and pair-level error attributes (`vx_error`, `vy_error`, `v_error`). These errors are stable-surface statistics, not spatially varying error rasters. They may be unavailable when too few valid stable pixels exist; providing a prior velocity field enables residual-based estimation on the stable mask.

## Temporal velocity composites

### Accepted input layouts

The compositing module recursively scans **single-band `.tif` files**, not the tracking NetCDFs or three-band GeoTIFFs.

**ITS_LIVE layout:** image-pair stems with supported Sentinel-1, Sentinel-2, or Landsat 8/9 scene naming, followed by `_vx.tif`, `_vy.tif`, and `_v.tif`. Optional error companions are `_v_error.tif`, `_vx_error.tif`, and `_vy_error.tif`. Generic date-only stems are not supported by this scanner. Native ITS_LIVE NetCDF products must first be exported into this raster layout.

**GAMMA layout:**

```text
20241013-20241025-Vx.tif
20241013-20241025-Vy.tif
20241013-20241025-V.tif
```

The GAMMA loader accepts these velocity triplets without error files and estimates errors during processing.

For example, a local Sentinel-1 autoRIFT GeoTIFF can be split into the GAMMA-compatible layout using GDAL command-line tools. Create `/path/to/velocity_triplets` first:

```bash
gdal_translate -b 1 /path/to/velocity_20241013_20241025.tif /path/to/velocity_triplets/20241013-20241025-Vx.tif
gdal_translate -b 2 /path/to/velocity_20241013_20241025.tif /path/to/velocity_triplets/20241013-20241025-Vy.tif
gdal_translate -b 3 /path/to/velocity_20241013_20241025.tif /path/to/velocity_triplets/20241013-20241025-V.tif
```

This is a file-format adapter, not GAMMA processing. The loader labels all such records `GAMMA_S1` and does not import the autoRIFT NetCDF error attributes. Optical or NISAR records require a suitable metadata/ingestion adaptation to preserve sensor identity; do not interpret this label as their actual sensor.

Velocity components must be expressed in a common vector basis before combination. The current compositing code reprojects rasters but does **not** rotate velocity vectors between map projections.

### Run composites

From the repository root:

```bash
cd 2_velocity_pipline_seasonally_monthly_yearly
python run_pipeline.py --help
```

For local velocity triplets only:

```bash
python run_pipeline.py --gamma-dir /path/to/velocity_triplets --no-itslive --output-dir /path/to/composites --bbox-latlon 150.0 -69.0 154.0 -67.0 --resolution 120 --modes monthly seasonal yearly --no-iqr-filter
```

The example bounding box covers a Cook-area domain. Replace it with bounds appropriate to your imagery and study area. Exactly one bounding-box option is required: `--bbox-latlon LON_MIN LAT_MIN LON_MAX LAT_MAX` or `--bbox-3031 X_MIN Y_MIN X_MAX Y_MAX`.

For both supported data sources:

```bash
python run_pipeline.py --itslive-dir /path/to/itslive_rasters --gamma-dir /path/to/gamma_rasters --output-dir /path/to/composites --bbox-latlon 150.0 -69.0 154.0 -67.0 --resolution 120 --modes monthly --date-start 2022-01-01 --date-end 2025-12-31
```

Both data sources are enabled by default. Use `--no-gamma` for ITS_LIVE-only processing or `--no-itslive` for GAMMA-layout-only processing. Each directory option accepts multiple paths.

Supported temporal modes are `monthly`, `seasonal`, `yearly`, `all`, and `fixed_6`, `fixed_12`, `fixed_18`, `fixed_30`, `fixed_60`. Records are assigned by the **midpoint of the image-pair acquisition dates**, not by overlap weighting across calendar bins. Fixed windows begin at the earliest retained midpoint. Antarctic seasons are December–February (summer), March–May (autumn), June–August (winter), and September–November (spring).

For the Python API, edit `example.py`. Configuration fields are `itslive_dirs` and `gamma_dirs` (plural, lists of paths).

### Filtering and averaging

Processing follows this order:

1. Read, clip, and reproject each velocity triplet to the common grid; obtain or estimate uncertainties.
2. Apply single-scene spatial MAD filtering and optional Universal Median Test filtering.
3. Exclude records below the selected coverage threshold.
4. Apply temporal MAD and optional IQR filtering within each group.
5. Compute inverse-variance-weighted means and export results.

The CLI defaults enable spatial and temporal MAD filtering, disable UMT and IQR filtering, use a 120 m EPSG:3031 target grid, and limit ITS_LIVE baselines to 33 days. **The Python dataclass enables IQR by default**, unlike the CLI; set `use_iqr_filter` explicitly when reproducing a command-line run. Filter windows are specified in pixels, so their physical footprint changes with grid spacing.

For a velocity component or input speed uᵢ with uncertainty σᵢ, multi-record averaging uses:

$$
w_i = \frac{1}{\sigma_i^2}, \qquad
\bar{u} = \frac{\sum_i w_i u_i}{\sum_i w_i}, \qquad
\sigma_{\bar{u}} = \left(\sum_i w_i\right)^{-1/2}.
$$

The effective observation count is:

$$
N_{\mathrm{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
$$

The module also computes a synthesized speed from the averaged components:

$$
V_{\mathrm{synth}} = \sqrt{\bar{V}_x^2 + \bar{V}_y^2}.
$$

`v` is the weighted mean of input speed magnitudes; `v_synth` is the magnitude of the averaged vector components. They need not be equal. Synthesized-speed uncertainty is propagated from component uncertainties without a covariance term. The `all` mode is an uncertainty-weighted composite, not a duration-weighted mean or a time-series inversion.

### Composite outputs

```text
composites/
├── scene_inventory.csv
├── grid_info.json
├── master_summary.csv
└── monthly/
    ├── summary_monthly.csv
    ├── group_log_monthly.txt
    ├── timeseries_summary.png           # Optional
    └── 2024-10/
        ├── 2024-10_vx.tif
        ├── 2024-10_vy.tif
        ├── 2024-10_v.tif
        ├── 2024-10_vx_err.tif
        ├── 2024-10_vy_err.tif
        ├── 2024-10_v_err.tif
        ├── 2024-10_v_synth.tif
        ├── 2024-10_v_synth_err.tif
        ├── 2024-10_neff.tif             # Optional
        └── 2024-10_overview.png         # Optional
```

Other temporal modes have the same layout. Composite rasters are single-band Float32 GeoTIFFs with NaN NoData. Velocities and uncertainties are in m yr⁻¹; `neff` is dimensionless and is saved from the direct-speed weights. Use `--no-plots` or `--no-neff` to disable optional outputs.

## Processing guidance

Select tracking and filtering parameters for the motion, texture, and data coverage of your study area. In particular, review speed limits before applying a configuration to fast-flowing ice, and interpret matching-window size separately from output grid spacing.

For reliable and reproducible processing:

- Use dedicated per-pair output directories and retain intermediate files when inspecting results.
- Separate batch runs for different footprints or sensor combinations sharing acquisition dates, because date-based pair names alone do not distinguish them.
- Review the compositing NoData setting if genuine zero velocities or components matter: `PipelineConfig.no_data_value` defaults to `0.0`.
- Inspect estimated uncertainties before weighting records. Multi-record weighted averages require finite, nonzero uncertainties, and their formal uncertainty does not include correlations between image pairs.
- Preserve acquisition IDs, auxiliary-data provenance, software versions, configurations, pair inventories, and processing logs alongside the outputs.

## Contributing and support

Questions, bug reports, and workflow improvements can be shared through the repository's GitHub issues and pull requests. For processing issues, include the module and sensor used, product type, software versions, relevant configuration settings, and a minimal example or error log. Remove private paths, credentials, and sensitive data before sharing.

Useful contributions include additional data readers, sensor-aware metadata handling, runnable examples, environment specifications, and tests for tracking and compositing workflows.

## Research context and citation

LHVS supports the satellite velocity analysis associated with *Oceanic and Bathymetric Controls on the Kinematic Evolution and Structural Destabilization of Cook Ice Shelf*. When using the project in research, acknowledge LHVS and cite the software and datasets contributing to your analysis. Use the associated manuscript's bibliographic details when available.

For the autoRIFT tracking method, cite:

Lei, Y., Gardner, A.S., Agram, P., 2021. Autonomous Repeat Image Feature Tracking (autoRIFT) and Its Application for Tracking Ice Displacement. *Remote Sensing* 13, 749. [https://doi.org/10.3390/rs13040749](https://doi.org/10.3390/rs13040749).

LHVS builds on autoRIFT, Geogrid, COMPASS, ISCE3, and HyP3. Their development teams retain credit for the underlying methods and software. Follow the applicable upstream citation guidance and retain attribution notices for adapted code.

## License

A project-level license has not yet been declared. Upstream software and datasets remain subject to their respective licenses and usage conditions.
