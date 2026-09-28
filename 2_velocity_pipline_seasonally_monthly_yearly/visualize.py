"""
visualize.py
============
Save PNG overview figures for each processed group.
No plt.show() is ever called – all outputs go to disk.

Produces per-group panels:
  • V (speed) with quiver arrows from Vx/Vy
  • Vx, Vy
  • V_error  (or V_synth_error)
  • N_eff    (effective observation count)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _setup_mpl():
    """Configure matplotlib backend before any import."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend – never opens windows
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return plt, mcolors


def save_group_figure(
    group_name: str,
    result: dict,
    out_dir: str | Path,
    target_transform,
    target_crs_str: str = "EPSG:3031",
    dpi: int = 150,
    vmax: Optional[float] = None,
    quiver_stride: int = 20,
) -> None:
    """
    Create and save a 2×3 overview panel for one temporal group.

    Parameters
    ----------
    group_name       : label for titles / filename
    result           : dict from weighted_avg.process_stack()
    out_dir          : directory to write the PNG
    target_transform : rasterio Affine (used to build extent)
    target_crs_str   : CRS string for axis labels
    dpi              : figure resolution
    vmax             : colour-scale maximum for speed maps (None = auto)
    quiver_stride    : sub-sampling stride for quiver arrows
    """
    plt, mcolors = _setup_mpl()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v      = result.get("v_synth",     result.get("v"))
    v_err  = result.get("v_synth_err", result.get("v_err"))
    vx     = result["vx"]
    vy     = result["vy"]
    neff   = result.get("neff_v",  result.get("neff_vx"))

    H, W = v.shape

    # Build axis extent in EPSG-3031 km
    tf = target_transform
    x0 = tf.c / 1e3;  x1 = (tf.c + W * tf.a) / 1e3
    y1 = tf.f / 1e3;  y0 = (tf.f + H * tf.e) / 1e3   # e < 0
    extent = [x0, x1, y0, y1]

    vm = float(np.nanpercentile(v, 98)) if vmax is None else vmax
    vm = max(vm, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                              constrained_layout=True)
    fig.suptitle(f"Ice Velocity  |  {group_name}", fontsize=14, fontweight="bold")

    # ── V (speed) ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(v, origin="upper", extent=extent,
                   cmap="RdYlBu_r", vmin=0, vmax=vm, aspect="equal")
    plt.colorbar(im, ax=ax, label="Speed (m/yr)")
    ax.set_title("Speed  V  (m/yr)")

    # Quiver overlay
    s = quiver_stride
    ys = np.arange(0, H, s)
    xs = np.arange(0, W, s)
    yi, xi = np.meshgrid(ys, xs, indexing="ij")
    qx = tf.c / 1e3 + (xi + 0.5) * tf.a / 1e3
    qy = tf.f / 1e3 + (yi + 0.5) * tf.e / 1e3

    u = np.where(np.isfinite(vx[ys][:, xs]), vx[ys][:, xs], 0.0)
    w = np.where(np.isfinite(vy[ys][:, xs]), vy[ys][:, xs], 0.0)
    mag = np.sqrt(u**2 + w**2)
    mask = mag > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        un = np.where(mask, u / mag, 0.0)
        wn = np.where(mask, w / mag, 0.0)
    ax.quiver(qx[mask], qy[mask], un[mask], wn[mask],
              color="k", alpha=0.5, scale=60, width=0.003)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    # ── Vx ────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    vlim = float(np.nanpercentile(np.abs(vx), 98)) if np.any(np.isfinite(vx)) else 1.0
    vlim = max(vlim, 1.0)
    im = ax.imshow(vx, origin="upper", extent=extent,
                   cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    plt.colorbar(im, ax=ax, label="Vx (m/yr)")
    ax.set_title("Vx  (m/yr)")

    # ── Vy ────────────────────────────────────────────────────────────────────
    ax = axes[0, 2]
    vlim = float(np.nanpercentile(np.abs(vy), 98)) if np.any(np.isfinite(vy)) else 1.0
    vlim = max(vlim, 1.0)
    im = ax.imshow(vy, origin="upper", extent=extent,
                   cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    plt.colorbar(im, ax=ax, label="Vy (m/yr)")
    ax.set_title("Vy  (m/yr)")

    # ── V error ───────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    if v_err is not None and np.any(np.isfinite(v_err)):
        e99 = float(np.nanpercentile(v_err, 99))
        e99 = max(e99, 1.0)
        im = ax.imshow(v_err, origin="upper", extent=extent,
                       cmap="YlOrRd", vmin=0, vmax=e99, aspect="equal")
        plt.colorbar(im, ax=ax, label="σV (m/yr)")
        ax.set_title("Speed Error  σV  (m/yr)")
    else:
        ax.set_visible(False)

    # ── N_eff ─────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    if neff is not None and np.any(np.isfinite(neff)):
        n99 = float(np.nanpercentile(neff, 99))
        n99 = max(n99, 1.0)
        im = ax.imshow(neff, origin="upper", extent=extent,
                       cmap="Blues", vmin=0, vmax=n99, aspect="equal")
        plt.colorbar(im, ax=ax, label="N_eff")
        ax.set_title("Effective Obs. Count  N_eff")
    else:
        ax.set_visible(False)

    # ── V_synth vs V_direct scatter (text summary) ───────────────────────────
    ax = axes[1, 2]
    v_direct = result.get("v")
    if v_direct is not None and np.any(np.isfinite(v_direct)) \
            and np.any(np.isfinite(result["v_synth"])):
        diff = result["v_synth"] - v_direct
        valid_diff = diff[np.isfinite(diff)]
        ax.hist(valid_diff, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("V_synth − V_direct (m/yr)")
        ax.set_ylabel("Pixel count")
        ax.set_title("Speed Residual\n(synthesis vs direct average)")
        mn   = float(np.nanmean(valid_diff))
        rmse = float(np.sqrt(np.nanmean(valid_diff**2)))
        ax.text(0.97, 0.97, f"mean={mn:.1f}\nRMSE={rmse:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    else:
        ax.set_visible(False)

    safe_name = group_name.replace("/", "-").replace(" ", "_")
    out_path  = out_dir / f"{safe_name}_overview.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot saved → %s", out_path.name)


def save_timeseries_summary(
    summary_records: list,
    output_dir: str | Path,
    dpi: int = 150,
) -> None:
    """
    After all groups are processed, plot mean-speed time series per mode.

    summary_records : list of dicts from the pipeline run_group() results
    """
    if not summary_records:
        return

    plt, _ = _setup_mpl()
    import pandas as pd

    out_dir = Path(output_dir)
    df = pd.DataFrame(summary_records)

    if "mean_date" not in df.columns or df.empty:
        return

    df = df.sort_values("mean_date")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                              constrained_layout=True)
    fig.suptitle("Mean Ice Velocity Time Series", fontsize=13, fontweight="bold")

    for ax, col, label, color in zip(
        axes,
        ["mean_v", "mean_vx", "mean_vy"],
        ["V (m/yr)", "Vx (m/yr)", "Vy (m/yr)"],
        ["steelblue", "tomato", "seagreen"],
    ):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        valid = df[col].notna()
        ax.plot(df.loc[valid, "mean_date"], df.loc[valid, col],
                "o-", color=color, ms=4, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Date")
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    out_path = out_dir / "timeseries_summary.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Time-series plot → %s", out_path)
