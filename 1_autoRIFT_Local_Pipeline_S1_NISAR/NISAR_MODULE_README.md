# NISAR 模块 — autoRIFT_SAR_Local_Pipeline v4 补丁说明

## 交付文件

| 文件 | 说明 |
|---|---|
| `src/nisar_processor.py` | ✨ **新增** — NISAR 核心处理模块 |
| `src/geogrid_builder_patch.py` | 📝 说明文件 — `run_geogrid_sar()` 的改动说明（非独立文件，直接修改原 `geogrid_builder.py`）|
| `run_pipeline.py` | 🔄 **替换** — 增加传感器分发逻辑，S1 流程不变 |
| `config_nisar.yaml` | ✨ **新增** — NISAR 配置文件模板 |

---

## 与 hyp3_autoRIFT 官方 nisar_isce3.py 的对照

| 功能 | 官方 hyp3 | 本地管线（v4）|
|---|---|---|
| 产品下载 | `asf_search.download()` | ❌ 移除，用户提供本地 `.h5` |
| RSLC 协同配准 | `run_isce3()` → rdr2geo/geo2rdr/resample_slc | `nisar_processor.run_isce3_coreg()` ✅ 完整保留 |
| 幅度提取 | `convert_slc_to_uint8_amplitude()` | `nisar_processor.extract_amplitude()` ✅ 逻辑等价 |
| GSLC 裁剪 | `crop_gslcs()` | `nisar_processor.crop_gslcs_overlap()` ✅ 等价 |
| RSLC 元数据 | `loadMetadataRslc()` + `mock_s1_orbit_file()` | ✅ 完整移植（`_mock_s1_orbit_file()`）|
| GSLC 元数据 | `GSLCMetadata` class | ✅ 完整移植 |
| Geogrid | `runGeogrid(optical_flag=0/1)` | `geogrid_builder.run_geogrid_sar(optical_flag=)` ✅ |
| autoRIFT | `generateAutoriftProduct()` | `autorift_runner.run_autorift()` ✅（本地管线自己的实现）|
| DEM | `hyp3lib.dem.prepare_dem_geotiff()` 自动下载 | ❌ 移除，从 config `dem_file` 读取本地文件 |
| 参数文件 | JPL shapefile + `find_jpl_parameter_info()` | 使用 SPS 参数文件或均匀回退（与 S1 路径完全一致）|

---

## 如何应用补丁

### 1. 复制新文件
```bash
cp src/nisar_processor.py  <your_pipeline>/src/
```

### 2. 修改 `geogrid_builder.py`（2 处改动）

在 `run_geogrid_sar()` 函数签名末尾增加参数：
```python
# 原来
def run_geogrid_sar(meta_r, meta_s, epsg, geogrid_inputs, working_dir="."):

# 改为
def run_geogrid_sar(meta_r, meta_s, epsg, geogrid_inputs, working_dir=".",
                    optical_flag: int = 0):   # ← 新增
```

在函数体内，将 `runGeogrid()` 调用中的 `optical_flag=0` 改为：
```python
geogrid_run_info = runGeogrid(
    info=meta_r,
    info1=meta_s,
    optical_flag=optical_flag,   # ← 原来是硬编码 0
    epsg=epsg,
    **geogrid_inputs_abs,
)
```

### 3. 替换 `run_pipeline.py`
用交付的 `run_pipeline.py` 替换原文件（S1 逻辑完全不变，增加 NISAR 分支）。

### 4. 运行 NISAR 处理
```bash
# 使用 NISAR 配置文件
python run_pipeline.py --config config_nisar.yaml

# 保留中间文件（调试用）
python run_pipeline.py --config config_nisar.yaml --keep-intermediates

# 详细日志
python run_pipeline.py --config config_nisar.yaml --verbose
```

---

## 传感器自动检测逻辑

`run_pipeline.py` 通过 config 字段自动判断传感器：

```
config.input.reference_h5  存在 → NISAR
config.input.reference_zip 存在 → Sentinel-1（向后兼容，行为不变）
```

---

## NISAR 处理流程（v4）

```
NISAR .h5 文件
    ↓
[1] detect_nisar_type()     → RSLC 或 GSLC
[2] detect_polarization()   → HH / VV（优先 HH）
    ↓
┌─────────────────────────────────┬───────────────────────────────────┐
│ RSLC（斜距几何）                  │ GSLC（地理编码）                    │
│ run_isce3_coreg()               │ crop_gslcs_overlap()              │
│  → rdr2geo                     │  → 裁剪至空间重叠区                  │
│  → geo2rdr                     │                                   │
│  → resample_slc                │                                   │
│ extract_amplitude(ref HDF5)    │ extract_amplitude(ref tif)        │
│ extract_amplitude(coreg sec)   │ extract_amplitude(sec tif)        │
│ loadMetadataRslc + mock orbit  │ GSLCMetadata                      │
│ optical_flag = 0               │ optical_flag = 1                  │
└─────────────────────────────────┴───────────────────────────────────┘
    ↓                                         ↓
[5] build_geogrid_inputs()  （与 S1 路径相同，SPS 文件支持）
[6] run_geogrid_sar(optical_flag=0/1)
[7] run_autorift()          （与 S1 完全相同）
    ↓
velocity_{pair}.tif / velocity_{pair}.nc
```

---

## 依赖说明

NISAR 处理需要额外安装（仅 NISAR 模式时）：

```bash
# ISCE3 Python 接口（协同配准 RSLC 必需）
conda install -c conda-forge isce3

# OpenCV（幅度图 HPS 滤波，本地管线已有 cv2 依赖）
# 已满足
```

S1 处理无需额外依赖，向后兼容。
