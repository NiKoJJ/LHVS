# LHVS

[English](README.md) | 简体中文

基于本地光学与 SAR 卫星影像的冰体水平速度制图与时间合成。

LHVS 提供一组 Python 工作流，用于从本地卫星影像中提取冰川和冰架速度场，并将影像对观测合成为不同时间尺度的速度产品。项目整合 autoRIFT 特征追踪、可配置的地理空间处理、异常值过滤和不确定性加权平均，支持冰体运动及其变化研究。

项目包含 Landsat 和 Sentinel-2 光学影像追踪、Sentinel-1 和 NISAR SAR 影像追踪，以及面向兼容速度栅格的独立时间合成模块。用户可处理单个影像对、筛选并批量处理影像对，或生成月度、季节、年度和固定时间窗口的速度图。

LHVS 最初用于东南极 Cook 冰架研究。通过配置影像、DEM、匹配参数和分析范围，可将工作流应用于其他冰川或冰架区域。时间合成模块默认采用南极极地立体投影（EPSG:3031）和南极季节划分。

## 主要功能

- **本地影像处理：** 使用已下载影像及用户提供的辅助数据。
- **光学与 SAR 工作流：** 从 Landsat、Sentinel-2、Sentinel-1 和 NISAR 产品中提取速度场。
- **可配置特征追踪：** 设置 DEM 覆盖范围、先验速度、搜索范围、匹配窗口大小及输出选项。
- **批量处理：** 根据时间基线、日期和传感器相关条件筛选影像对。
- **质量控制：** 在时间合成前进行空间和时间异常值过滤。
- **时间合成：** 采用逆方差加权合成速度观测，并输出不确定性估计。
- **地理空间输出：** 生成 GeoTIFF 速度图、可选的追踪结果 NetCDF、CSV 汇总和快速预览图。

## 工作流程

| 模块 | 输入 | 处理过程 | 输出 |
| --- | --- | --- | --- |
| 光学影像追踪 | Landsat 或 Sentinel-2 波段影像及 DEM | 预处理、GeogridOptical 和 autoRIFT 匹配 | 影像对速度 GeoTIFF 及可选 NetCDF |
| SAR 影像追踪 | Sentinel-1 SLC ZIP 或 NISAR RSLC/GSLC HDF5 产品及 DEM | 配准或重叠区域准备、幅度提取、Geogrid 和 autoRIFT 匹配 | 影像对速度 GeoTIFF 及可选 NetCDF |
| 时间合成 | 符合受支持 ITS_LIVE 或 GAMMA 命名规则的单波段速度栅格 | 统一网格处理、质量控制和不确定性加权平均 | 时间合成速度场、不确定性图、汇总表和图件 |

各模块可独立运行。如需合成本地追踪结果，应先将三波段 GeoTIFF 转换为受支持的单波段文件布局，具体见[时间合成速度场](#temporal-velocity-composites)。

## 项目结构

```text
.
├── README.md
├── README_zh-CN.md
├── 1_autoRIFT_Local_Pipeline_Optical/
│   ├── run_pipeline.py                 # 处理一个光学影像对
│   ├── batch_run.py                    # 筛选并处理多个影像对
│   ├── config.yaml
│   └── src/
│       ├── optical_processor.py
│       ├── geogrid_builder_optical.py
│       ├── autorift_runner_optical.py
│       └── utils.py
├── 1_autoRIFT_Local_Pipeline_S1_NISAR/
│   ├── run_pipeline.py                 # 处理一个 Sentinel-1 或 NISAR 影像对
│   ├── batch_run.py
│   ├── generate_parameter_pack.py      # 准备 DEM/SPS 输入，不执行追踪
│   ├── config.yaml                     # Sentinel-1 配置
│   ├── config_nisar.yaml               # NISAR 配置
│   ├── NISAR_MODULE_README.md           # NISAR 模块说明
│   └── src/
│       ├── s1_processor.py
│       ├── nisar_processor.py
│       ├── geogrid_builder.py
│       ├── autorift_runner.py
│       ├── utils.py
│       └── schemas/s1_cslc_template.yaml
└── 2_velocity_pipline_seasonally_monthly_yearly/
    ├── run_pipeline.py                 # 时间合成命令行入口
    ├── example.py                      # Python 配置示例
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

请在所用模块的目录中运行命令。光学和 SAR 模块均包含名为 `src` 的包，应分别运行，并避免其导入路径相互混淆。

## 安装与环境准备

LHVS 使用 Python 科学计算和地理空间库，并依赖外部影像追踪与配准软件。请根据需要运行的模块准备依赖；时间合成模块无需安装完整的影像追踪软件栈。

| 组件 | 所需软件 |
| --- | --- |
| 光学影像追踪 | Python、NumPy、PyYAML、GDAL Python 接口（`osgeo`）、OpenCV（`cv2`）、autoRIFT 和 Geogrid |
| Sentinel-1 追踪 | 追踪依赖，以及 `hyp3_autorift`、COMPASS、`s1reader` 和兼容的 ISCE3 环境 |
| NISAR 追踪 | 追踪依赖，以及 `hyp3_autorift`、`h5py`、ISCE3 和 `nisar.products` / `nisar.workflows` Python 模块 |
| NetCDF 输出 | `xarray` 和 `netCDF4` |
| 时间合成 | NumPy、pandas、rasterio、SciPy、pyproj、Shapely 和 Matplotlib |

可从 Python 3.10 及以上版本开始配置环境。对于影像追踪，应选择相关地理空间与 SAR 依赖支持的 Python 版本。SAR 软件栈建议在 Linux 或 WSL 中运行。

创建用于**时间合成**的环境：

```bash
conda create -n lhvs -c conda-forge python=3.11 numpy pandas rasterio scipy pyproj shapely matplotlib
conda activate lhvs
```

对于影像追踪，请遵循以下上游项目的安装说明，并使用相互兼容的版本：

- [autoRIFT 和 Geogrid](https://github.com/nasa-jpl/autoRIFT)
- [HyP3 autoRIFT](https://github.com/ASFHyP3/hyp3-autorift)
- [COMPASS](https://github.com/opera-adt/COMPASS)
- [ISCE3](https://github.com/isce-framework/isce3)

追踪模块使用这些软件包的接口，包括 `hyp3_autorift.vend.testGeogrid`。选择版本时请检查 API 兼容性，并保留环境信息以支持可重复处理。

## 快速开始

根据输入数据选择工作流：

- **光学波段影像：** 从[光学影像对追踪](#optical-image-pair-tracking)开始。
- **Sentinel-1 或 NISAR 产品：** 从[Sentinel-1 与 NISAR 追踪](#sentinel-1-and-nisar-tracking)开始。
- **已有速度栅格：** 从[时间合成速度场](#temporal-velocity-composites)开始。

影像追踪需要本地卫星产品和覆盖研究区的投影坐标系 DEM，也可提供稳定地表掩膜、空间变化的先验速度或匹配参数栅格。Sentinel-1 处理还需要合适的轨道信息。这些数据需另行下载和准备，不随代码提供。

以下命令中的路径仅为示例。运行前请替换为实际数据和输出路径，并调整研究区边界与处理参数。

<a id="optical-image-pair-tracking"></a>

## 光学影像对追踪

主处理流程识别 Landsat 4/5/7/8/9 和 Sentinel-2 文件名。常用默认波段为 Landsat 7/8/9 的 B8、Landsat 4/5 的 B2，以及 Sentinel-2 的 B08。批量处理时可明确指定其他波段。

### 配置并处理单个影像对

从项目根目录进入光学模块：

```bash
cd 1_autoRIFT_Local_Pipeline_Optical
```

编辑 `config.yaml`，至少将 `input.reference_path`、`input.secondary_path` 和 `input.dem_file` 替换为实际存在的文件。检查其他本地路径，包括 `input.ssm_file` 和所有 `sps_params` 项。没有对应文件的可选项应设为 `null`。

主要配置项包括：

- `processing.base_results_dir`：输出根目录，每个影像对使用独立子目录。
- `processing.epsg`：目标投影；设为 `null` 时使用 DEM 投影。
- `processing.chip_size_min_m` / `chip_size_max_m`：以米为单位的匹配窗口大小，并非输出像元间距。
- `processing.search_range_m_yr`：没有空间搜索参数时使用的搜索范围默认值。
- `processing.velocity_max_m_yr`：用于剔除不合理结果的速度上限。
- `processing.output_netcdf` 和 `save_offsets`：可选产品输出。
- `sps_params`：可选的先验速度、搜索范围、匹配窗口大小和地表坡度参数。

请使用合适的投影 DEM，并设置研究区范围。保留可识别的场景文件名，因为采集日期通过文件名解析。

```bash
python run_pipeline.py --config config.yaml --keep-intermediates
```

`--keep-intermediates` 保留参数栅格、偏移量和诊断文件。不提供该选项时，光学模块在成功处理后会删除影像对目录内除 `velocity_*.tif` 和 `velocity_*.nc` 外的所有内容。请使用专用输出目录，不要与无关文件混放。

### 批量处理

以下示例使用 Landsat B5 波段，最大时间基线为 60 天，最小空间重叠比例为 40%。路径为占位示例，请在运行前替换。

```bash
python batch_run.py --scene-dir /path/to/landsat --output-dir /path/to/optical_results --platform landsat --bands B5 --pipeline-dir . --base-config config.yaml --interval 60 --min-overlap 0.4 --dry-run
```

检查影像对列表后，移除 `--dry-run` 即可执行处理。对于 Sentinel-2，使用 `--platform s2 --bands B08` 并指定相应的影像目录。

光学模块的 `--interval` 指定**最大时间基线**，而非精确的重访间隔。脚本在该上限内，结合 `--min-days`、重叠比例、日期和传感器筛选条件选择符合要求的影像对。这一配对步骤不执行 SBAS 位移反演。

光学批量处理时，请将 `processing.base_results_dir` 和 `--output-dir` 设为相同的**绝对路径**：配置文件存放和完成状态检查使用 `--output-dir`，而追踪程序使用 `base_results_dir`。如需保留诊断文件，请对单个已生成的影像对配置运行 `--keep-intermediates`；光学批量命令行不会向追踪程序传递该选项。

本文档介绍的光学工作流适用于 Landsat 和 Sentinel-2。额外的 `planetscope` 和 `custom` 批量影像发现选项需要配套的下游文件名、日期和元数据处理支持。

<a id="sentinel-1-and-nisar-tracking"></a>

## Sentinel-1 与 NISAR 追踪

返回项目根目录后，进入 SAR 模块：

```bash
cd 1_autoRIFT_Local_Pipeline_S1_NISAR
```

### Sentinel-1

编辑 `config.yaml`，设置以下内容：

- `input.reference_zip` 和 `secondary_zip`：本地 Sentinel-1 SLC ZIP 文件。
- `input.dem_file`：本地 DEM；也可提供稳定地表掩膜。
- `orbit.orbit_dir`：轨道文件目录；`orbit.allow_download` 控制是否允许下载缺失的轨道文件。
- `burst_selection`：burst 选择参数；处理前请检查可用 burst。
- `processing` 和 `sps_params`：输出、匹配、先验速度及空间参数配置。

```bash
python run_pipeline.py --config config.yaml --list-bursts
python run_pipeline.py --config config.yaml --keep-intermediates
```

批量处理示例：

```bash
python batch_run.py --sensor s1 --data-dir /path/to/s1_slc --orbit-dir /path/to/orbits --output-dir /path/to/s1_results --pipeline-dir . --base-config config.yaml --interval 12 --tolerance 2 --dry-run
```

移除 `--dry-run` 可处理已选择的影像对；需要时添加 `--keep-intermediates`。与光学配对不同，SAR 模块的 `--interval` 是**目标时间基线**，允许在 `--tolerance` 范围内偏离该值，并不会枚举所有可能的时间基线。

### NISAR

编辑 `config_nisar.yaml`，包括 `input.reference_h5`、`secondary_h5`、`dem_file`、`nisar.frequency`，以及输出路径和可选空间参数路径。

代码按产品类型执行不同流程：

- **RSLC：** ISCE3 配准、幅度提取和雷达几何元数据准备。
- **GSLC：** 重叠区域裁剪、幅度提取和地理编码元数据准备。

```bash
python run_pipeline.py --config config_nisar.yaml --keep-intermediates
```

批量检查影像对：

```bash
python batch_run.py --sensor nisar --data-dir /path/to/nisar_h5 --output-dir /path/to/nisar_results --pipeline-dir . --base-config config_nisar.yaml --interval 12 --tolerance 2 --dry-run
```

`processing.output_resolution_m` 控制 SAR 流程中 Geogrid 使用的工作 DEM 网格间距。它与匹配窗口大小不同，也不能单独决定速度观测的有效空间分辨率。

仅准备 DEM/SPS 输入：

```bash
python generate_parameter_pack.py --config config_nisar.yaml --output /path/to/parameter_pack
```

该工具不生成与传感器相关的 `window_*.tif` 文件或速度产品；这些结果需要后续 Geogrid 和追踪步骤。

与光学模块一样，SAR 程序默认清理中间文件，除非指定 `--keep-intermediates`。SAR 清理保留速度 GeoTIFF、速度 NetCDF 和 YAML 配置文件，但会删除影像对目录中的其他文件及子目录。

### 追踪产品定义

两种追踪工作流均输出 Float32 类型的 GeoTIFF 文件 `velocity_<pair_name>.tif`：

| 波段 | 含义 | 单位 |
| --- | --- | --- |
| 1 | `Vx_m_yr`：沿输出投影 x 轴方向的速度 | m yr⁻¹ |
| 2 | `Vy_m_yr`：沿输出投影 y 轴方向的速度 | m yr⁻¹ |
| 3 | `V_m_yr`：速率，√(Vx² + Vy²) | m yr⁻¹ |

追踪 GeoTIFF 的 NoData 值为 `-32767`。投影坐标系的 x/y 速度分量不能直接视为地理坐标系的东向/北向分量。

可选的 `offset_<pair_name>.tif` 包含四个波段：`DX_px`、`DY_px`、`InterpMask` 和 `ChipSizeX`。可选 NetCDF 文件包含速度场、坐标、投影信息、匹配诊断及影像对级别的误差属性（`vx_error`、`vy_error`、`v_error`）。这些误差来自稳定地表统计，并非空间变化的误差栅格。有效稳定像元过少时，误差可能无法估计；提供先验速度场可支持基于稳定掩膜内残差的估计。

<a id="temporal-velocity-composites"></a>

## 时间合成速度场

### 支持的输入布局

时间合成模块递归扫描**单波段 `.tif` 文件**，不直接读取追踪 NetCDF 或三波段 GeoTIFF。

**ITS_LIVE 布局：** 文件主名采用受支持的 Sentinel-1、Sentinel-2 或 Landsat 8/9 场景对命名方式，分别添加 `_vx.tif`、`_vy.tif` 和 `_v.tif` 后缀。可选误差文件后缀为 `_v_error.tif`、`_vx_error.tif` 和 `_vy_error.tif`。扫描器不支持仅包含日期的通用文件主名。原生 ITS_LIVE NetCDF 产品需先转换为此类栅格布局。

**GAMMA 布局：**

```text
20241013-20241025-Vx.tif
20241013-20241025-Vy.tif
20241013-20241025-V.tif
```

GAMMA 读取器支持没有配套误差文件的上述速度三元组，并在处理时估计误差。

例如，可使用 GDAL 命令行工具将本地 Sentinel-1 autoRIFT GeoTIFF 拆分为兼容 GAMMA 布局的文件。请先创建 `/path/to/velocity_triplets` 目录：

```bash
gdal_translate -b 1 /path/to/velocity_20241013_20241025.tif /path/to/velocity_triplets/20241013-20241025-Vx.tif
gdal_translate -b 2 /path/to/velocity_20241013_20241025.tif /path/to/velocity_triplets/20241013-20241025-Vy.tif
gdal_translate -b 3 /path/to/velocity_20241013_20241025.tif /path/to/velocity_triplets/20241013-20241025-V.tif
```

这只是文件格式适配，不是 GAMMA 处理。读取器将所有此类记录标记为 `GAMMA_S1`，且不会导入 autoRIFT NetCDF 的误差属性。光学或 NISAR 记录需要相应的元数据或读取适配，才能保留真实传感器身份；不能将该标签视为其实际传感器。

合成前，速度分量必须表达在共同的矢量坐标基准下。当前代码会对栅格进行重投影，但**不会**在不同地图投影之间旋转速度矢量。

### 运行时间合成

从项目根目录进入时间合成模块：

```bash
cd 2_velocity_pipline_seasonally_monthly_yearly
python run_pipeline.py --help
```

仅处理本地速度三元组：

```bash
python run_pipeline.py --gamma-dir /path/to/velocity_triplets --no-itslive --output-dir /path/to/composites --bbox-latlon 150.0 -69.0 154.0 -67.0 --resolution 120 --modes monthly seasonal yearly --no-iqr-filter
```

示例边界框覆盖 Cook 附近区域。请根据影像覆盖和研究范围调整。必须且只能指定一种边界框选项：`--bbox-latlon LON_MIN LAT_MIN LON_MAX LAT_MAX` 或 `--bbox-3031 X_MIN Y_MIN X_MAX Y_MAX`。

同时使用两种受支持的数据来源：

```bash
python run_pipeline.py --itslive-dir /path/to/itslive_rasters --gamma-dir /path/to/gamma_rasters --output-dir /path/to/composites --bbox-latlon 150.0 -69.0 154.0 -67.0 --resolution 120 --modes monthly --date-start 2022-01-01 --date-end 2025-12-31
```

默认启用两种数据来源。仅处理 ITS_LIVE 时使用 `--no-gamma`；仅处理 GAMMA 布局时使用 `--no-itslive`。每个目录选项可接受多个路径。

支持的时间模式包括 `monthly`、`seasonal`、`yearly`、`all`，以及 `fixed_6`、`fixed_12`、`fixed_18`、`fixed_30`、`fixed_60`。记录按**影像对采集日期的中点**分组，而非根据与日历时间区间的重叠进行加权分配。固定窗口从保留记录的最早日期中点开始。南极季节划分为 12–2 月（夏季）、3–5 月（秋季）、6–8 月（冬季）和 9–11 月（春季）。

如使用 Python API，请编辑 `example.py`。配置字段为 `itslive_dirs` 和 `gamma_dirs`（复数形式，值为路径列表）。

### 过滤与平均

处理顺序如下：

1. 读取、裁剪并将每组速度三元组重投影至统一网格，获取或估计不确定性。
2. 应用单景空间 MAD 过滤，以及可选的通用中值检验（UMT）过滤。
3. 剔除覆盖比例低于指定阈值的记录。
4. 在各时间分组内应用时间 MAD 和可选的 IQR 过滤。
5. 计算逆方差加权平均并输出结果。

命令行默认启用空间和时间 MAD 过滤，关闭 UMT 和 IQR 过滤，使用 120 m 的 EPSG:3031 目标网格，并将 ITS_LIVE 时间基线限制为 33 天。**Python dataclass 默认启用 IQR**，与命令行默认值不同；复现命令行处理时，请明确设置 `use_iqr_filter`。过滤窗口以像元为单位，因此其对应的实际空间范围随网格间距变化。

对于不确定性为 σᵢ 的速度分量或输入速率 uᵢ，多条记录的加权平均为：

$$
w_i = \frac{1}{\sigma_i^2}, \qquad
\bar{u} = \frac{\sum_i w_i u_i}{\sum_i w_i}, \qquad
\sigma_{\bar{u}} = \left(\sum_i w_i\right)^{-1/2}.
$$

有效观测数量为：

$$
N_{\mathrm{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
$$

模块还根据平均速度分量计算合成速率：

$$
V_{\mathrm{synth}} = \sqrt{\bar{V}_x^2 + \bar{V}_y^2}.
$$

`v` 是输入速率大小的加权平均；`v_synth` 是平均矢量分量的模，二者不一定相等。合成速率的不确定性由分量不确定性传播得到，不包含协方差项。`all` 模式生成不确定性加权合成结果，并非按时段长度加权的平均值，也不执行时间序列反演。

### 合成产品输出

```text
composites/
├── scene_inventory.csv
├── grid_info.json
├── master_summary.csv
└── monthly/
    ├── summary_monthly.csv
    ├── group_log_monthly.txt
    ├── timeseries_summary.png           # 可选
    └── 2024-10/
        ├── 2024-10_vx.tif
        ├── 2024-10_vy.tif
        ├── 2024-10_v.tif
        ├── 2024-10_vx_err.tif
        ├── 2024-10_vy_err.tif
        ├── 2024-10_v_err.tif
        ├── 2024-10_v_synth.tif
        ├── 2024-10_v_synth_err.tif
        ├── 2024-10_neff.tif             # 可选
        └── 2024-10_overview.png         # 可选
```

其他时间模式采用相同布局。合成栅格为单波段 Float32 GeoTIFF，NoData 为 NaN。速度和不确定性的单位为 m yr⁻¹；`neff` 无量纲，由直接速率平均所用权重计算并保存。可使用 `--no-plots` 或 `--no-neff` 关闭相应可选输出。

## 处理建议

请根据研究区运动特征、影像纹理和数据覆盖选择追踪及过滤参数。将配置应用于快速流动冰体前，应检查速度上限；同时应区分匹配窗口大小与输出网格间距。

为提高处理可靠性和可重复性：

- 为每个影像对使用专用输出目录，检查结果时保留中间文件。
- 对相同采集日期但不同覆盖范围或传感器组合分别组织批量运行，因为仅包含日期的影像对名称不能区分这些情况。
- 如果真实零速度或零分量对分析重要，请检查合成模块的 NoData 设置：`PipelineConfig.no_data_value` 默认值为 `0.0`。
- 加权前检查估计的不确定性。多条记录的加权平均需要有限且非零的不确定性，其形式不确定性不包含影像对之间的相关性。
- 与输出一起保存采集标识、辅助数据来源、软件版本、配置、影像对清单和处理日志。

## 贡献与支持

欢迎通过项目 GitHub issues 和 pull requests 提交问题、缺陷报告及工作流改进。反馈处理问题时，请提供使用的模块和传感器、产品类型、软件版本、相关配置，以及最小示例或错误日志。分享前请移除私人路径、访问凭据和敏感数据。

可贡献的内容包括新增数据读取器、保留传感器身份的元数据处理、可运行示例、环境配置说明，以及追踪和时间合成工作流的测试。

## 研究背景与引用

LHVS 支持论文 *Oceanic and Bathymetric Controls on the Kinematic Evolution and Structural Destabilization of Cook Ice Shelf* 中的卫星速度分析。在研究中使用本项目时，请注明 LHVS，并引用分析所用的软件和数据集。相关论文的正式书目信息公布后，可据此引用。

使用 autoRIFT 追踪方法时，请引用：

Lei, Y., Gardner, A.S., Agram, P., 2021. Autonomous Repeat Image Feature Tracking (autoRIFT) and Its Application for Tracking Ice Displacement. *Remote Sensing* 13, 749. [https://doi.org/10.3390/rs13040749](https://doi.org/10.3390/rs13040749).

LHVS 基于 autoRIFT、Geogrid、COMPASS、ISCE3 和 HyP3 构建。底层方法与软件的贡献归属于相应开发团队。请遵循上游项目的引用要求，并保留所改编代码的署名声明。

## 许可证

本项目尚未声明项目级许可证。上游软件和数据集仍受各自许可证及使用条件约束。
