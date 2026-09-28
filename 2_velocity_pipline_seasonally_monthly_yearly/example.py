"""
Example Python entry point for the glacier velocity pipeline.

Edit the paths and bounding box below before running:

    python example.py
"""

from pathlib import Path

from config import PipelineConfig, TemporalMode, VxErrorMode
from pipeline import run_pipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent

    cfg = PipelineConfig(
        itslive_dirs=[str(project_root / "data" / "itslive")],
        gamma_dirs=[str(project_root / "data" / "gamma")],
        output_dir=str(project_root / "output_example"),
        bbox_latlon=(158.0, -68.5, 163.5, -66.5),
        bbox_3031=None,
        temporal_modes=[
            TemporalMode.MONTHLY,
            TemporalMode.SEASONAL,
        ],
        target_resolution=120.0,
        use_itslive=True,
        use_gamma=True,
        itslive_sensors=None,
        itslive_max_days=33,
        gamma_max_days=None,
        vx_error_mode=VxErrorMode.ISOTROPIC,
        gamma_error_mode=VxErrorMode.LOCAL_STD,
        use_spatial_mad=True,
        use_temporal_mad=True,
        use_iqr_filter=False,
        min_coverage_frac=0.0,
        min_valid_obs=1,
        save_plots=True,
        save_neff=True,
        n_workers=1,
        io_threads=4,
        log_level="INFO",
    )

    summaries = run_pipeline(cfg)

    if summaries:
        print(f"Pipeline completed successfully: {len(summaries)} groups processed.")
        print(f"Output directory: {cfg.output_dir}")
        for item in summaries[:5]:
            print(
                f"- {item['mode']} | {item['group']} | "
                f"records={item['n_records']}/{item['n_total']} | "
                f"coverage={item['coverage_pct']:.1f}%"
            )
    else:
        print("Pipeline produced no summaries. Check the input paths and logs.")


if __name__ == "__main__":
    main()
