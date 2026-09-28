"""
error_and_outlier.py
====================
1. Error estimation for Vx / Vy when only V_error is available (ITS_LIVE)
   or when no error files exist at all (GAMMA).

2. IQR-based per-pixel outlier removal (Yang et al. 2026, Section 2.1.2–2.1.3).
"""

from __future__ import annotations
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label, uniform_filter

log = logging.getLogger(__name__)


def _validate_umt_params(
    window_size: int,
    epsilon: float,
    threshold: float,
    min_segment_size: int,
) -> None:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("UMT window_size must be an odd integer >= 3")
    if epsilon < 0:
        raise ValueError("UMT epsilon must be >= 0")
    if threshold <= 0:
        raise ValueError("UMT threshold must be > 0")
    if min_segment_size < 1:
        raise ValueError("UMT min_segment_size must be >= 1")


def _build_neighbour_stack(data: np.ndarray, window_size: int) -> np.ndarray:
    """Return a stack of neighbours around each pixel, excluding centre."""
    H, W = data.shape
    pad = window_size // 2
    padded = np.pad(
        data.astype(np.float32),
        pad_width=pad,
        mode="constant",
        constant_values=np.nan,
    )

    neighbours = []
    for dy in range(window_size):
        for dx in range(window_size):
            if dy == pad and dx == pad:
                continue
            neighbours.append(padded[dy:dy + H, dx:dx + W])
    return np.stack(neighbours, axis=0).astype(np.float32, copy=False)


def _compute_umt_bad_mask(
    data: np.ndarray,
    window_size: int,
    epsilon: float,
    threshold: float,
) -> np.ndarray:
    """Boolean mask of pixels flagged by the Universal Median Test."""
    neighbours = _build_neighbour_stack(data, window_size)
    neigh_med = np.nanmedian(neighbours, axis=0).astype(np.float32)
    residuals = np.abs(neighbours - neigh_med[np.newaxis]).astype(np.float32)
    resid_med = np.nanmedian(residuals, axis=0).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(data - neigh_med) / (resid_med + np.float32(epsilon))

    return np.isfinite(data) & np.isfinite(neigh_med) & (score > threshold)


def apply_universal_median_test(
    vx: np.ndarray,
    vy: np.ndarray,
    v: Optional[np.ndarray] = None,
    vx_err: Optional[np.ndarray] = None,
    vy_err: Optional[np.ndarray] = None,
    v_err: Optional[np.ndarray] = None,
    window_size: int = 5,
    epsilon: float = 0.1,
    threshold: float = 5.0,
    min_segment_size: int = 25,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray], Dict[str, int]]:
    """
    Apply the Universal Median Test (UMT) to a single scene.

    Pixels are culled when the normalized median residual exceeds the
    threshold in either velocity component. After culling, small connected
    segments of remaining valid pixels can also be removed.
    """
    _validate_umt_params(window_size, epsilon, threshold, min_segment_size)

    vx_f = vx.copy()
    vy_f = vy.copy()
    v_f = None if v is None else v.copy()
    vx_err_f = None if vx_err is None else vx_err.copy()
    vy_err_f = None if vy_err is None else vy_err.copy()
    v_err_f = None if v_err is None else v_err.copy()

    bad_vx = _compute_umt_bad_mask(vx_f, window_size, epsilon, threshold)
    bad_vy = _compute_umt_bad_mask(vy_f, window_size, epsilon, threshold)
    bad_any = bad_vx | bad_vy

    arrays = [vx_f, vy_f, v_f, vx_err_f, vy_err_f, v_err_f]
    for arr in arrays:
        if arr is not None:
            arr[bad_any] = np.nan

    removed_outliers = int(np.sum(bad_any))

    removed_small_segments = 0
    valid_mask = np.isfinite(vx_f) & np.isfinite(vy_f)
    if min_segment_size > 1 and np.any(valid_mask):
        labels, n_labels = label(valid_mask, structure=np.ones((3, 3), dtype=np.int8))
        if n_labels > 0:
            sizes = np.bincount(labels.ravel())
            small = sizes < min_segment_size
            small[0] = False
            small_mask = small[labels]
            removed_small_segments = int(np.sum(small_mask & valid_mask))
            if removed_small_segments:
                for arr in arrays:
                    if arr is not None:
                        arr[small_mask] = np.nan

    stats = {
        "n_outlier_pixels": removed_outliers,
        "n_small_segment_pixels": removed_small_segments,
    }
    return vx_f, vy_f, v_f, vx_err_f, vy_err_f, v_err_f, stats

# ─────────────────────────────────────────────────────────────────────────────
# Local standard deviation (spatial smoothing)
# ─────────────────────────────────────────────────────────────────────────────

def local_std(
    data: np.ndarray,
    window_size: int = 5,
    min_valid: int = 4,
) -> np.ndarray:
    """
    Compute local standard deviation over a sliding window.
    Handles NaN by treating them as missing.

    Parameters
    ----------
    data        : 2D float32 array with NaN for missing
    window_size : side length of square window (pixels)
    min_valid   : minimum valid cells inside window to keep a result

    Returns
    -------
    std_arr : 2D float32 array, NaN where coverage < min_valid
    """
    valid = np.isfinite(data)
    filled = np.where(valid, data.astype(np.float64), 0.0)

    # Count of valid pixels per window
    count = uniform_filter(valid.astype(np.float64), size=window_size,
                           mode="constant", cval=0.0) * window_size**2

    mean  = uniform_filter(filled, size=window_size,
                           mode="constant", cval=0.0)
    mean2 = uniform_filter(filled**2, size=window_size,
                           mode="constant", cval=0.0)

    # Correct mean for partial windows
    cnt_raw = uniform_filter(valid.astype(np.float64), size=window_size,
                             mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_mean = np.where(cnt_raw > 0, mean / cnt_raw, np.nan)
        local_m2   = np.where(cnt_raw > 0, mean2 / cnt_raw, np.nan)

    var = local_m2 - local_mean**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var).astype(np.float32)

    # Mask pixels with too few neighbours
    std[count < min_valid] = np.nan

    return std


# ─────────────────────────────────────────────────────────────────────────────
# Vx / Vy error estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_vx_vy_errors(
    vx: np.ndarray,
    vy: np.ndarray,
    v_error: Optional[np.ndarray],
    mode: str = "isotropic",
    window_size: int = 5,
    min_valid: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-pixel errors for Vx and Vy.

    Parameters
    ----------
    vx, vy      : velocity components (float32, NaN = missing)
    v_error     : total speed error (float32) or None
    mode        : 'isotropic' | 'local_std' | 'proportional'
    window_size : only used for 'local_std'
    min_valid   : only used for 'local_std'

    Returns
    -------
    vx_err, vy_err : float32 arrays of same shape as vx
    """
    H, W = vx.shape

    if mode == "isotropic":
        # σVx = σVy = σV / √2
        if v_error is not None:
            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            vx_err = (v_error * inv_sqrt2).astype(np.float32)
            vy_err = (v_error * inv_sqrt2).astype(np.float32)
        else:
            # Fallback to local_std when no v_error exists
            vx_err = local_std(vx, window_size, min_valid)
            vy_err = local_std(vy, window_size, min_valid)

    elif mode == "proportional":
        # σVx = σV * |Vx| / V   (direction-aware)
        # Use a minimum V threshold to prevent blowup at tile edges where
        # bilinear reprojection can produce near-zero V values.
        if v_error is not None:
            v_mag = np.sqrt(vx**2 + vy**2)
            # Minimum threshold: 5 m/yr. Below this, fall back to isotropic.
            # This prevents vx_e = v_error / ~0 → inf at tile edges.
            v_safe = np.where(v_mag > 5.0, v_mag, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                vx_err = np.where(np.isfinite(v_safe),
                                  v_error * np.abs(vx) / v_safe,
                                  v_error / np.sqrt(2)).astype(np.float32)
                vy_err = np.where(np.isfinite(v_safe),
                                  v_error * np.abs(vy) / v_safe,
                                  v_error / np.sqrt(2)).astype(np.float32)
        else:
            vx_err = local_std(vx, window_size, min_valid)
            vy_err = local_std(vy, window_size, min_valid)

    elif mode == "local_std":
        vx_err = local_std(vx, window_size, min_valid)
        vy_err = local_std(vy, window_size, min_valid)

    else:
        raise ValueError(f"Unknown vx_error_mode: {mode}")

    # Ensure strictly positive (set zero/negative to NaN)
    vx_err[vx_err <= 1e-8] = np.nan
    vy_err[vy_err <= 1e-8] = np.nan

    return vx_err, vy_err


# ─────────────────────────────────────────────────────────────────────────────
# IQR outlier removal  (Yang et al. 2026)
# ─────────────────────────────────────────────────────────────────────────────

def _fast_q1q3(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast per-pixel Q1 / Q3 using np.partition (O(N) vs O(N log N) sort).

    For small N (≤ ~30 scenes) this is noticeably faster than nanpercentile;
    for large N the speedup is even greater.

    NaN values are handled by substituting a sentinel (max float32) before
    partitioning, which pushes NaNs to the end of the sorted view.
    """
    N, H, W = stack.shape

    # Replace NaN with +inf so they sort to the tail
    s = stack.copy()
    nan_mask = ~np.isfinite(s)
    s[nan_mask] = np.finfo(np.float32).max

    # Number of valid observations per pixel
    n_valid = N - nan_mask.sum(axis=0)  # (H, W)

    # Index positions for Q1 and Q3
    # np.partition along axis=0 gives the k-th smallest value at position k
    i_q1 = np.clip(((n_valid - 1) * 0.25).astype(int), 0, N - 1)
    i_q3 = np.clip(((n_valid - 1) * 0.75).astype(int), 0, N - 1)

    # Partial sort – only guarantees position k, which is all we need
    # We need both Q1 and Q3, so partition at the larger index
    i_max = i_q3.max()
    partitioned = np.partition(s, kth=min(i_max, N - 1), axis=0)

    # Gather Q1 and Q3 using advanced indexing over flat (H*W) space
    flat_idx = np.arange(H * W)
    q1 = partitioned.reshape(N, -1)[i_q1.ravel(), flat_idx].reshape(H, W)
    q3 = partitioned.reshape(N, -1)[i_q3.ravel(), flat_idx].reshape(H, W)

    # Pixels with no valid data → NaN
    no_data = n_valid == 0
    q1[no_data] = np.nan
    q3[no_data] = np.nan

    # Undo +inf sentinel for any q1/q3 that landed on a NaN slot
    q1[q1 == np.finfo(np.float32).max] = np.nan
    q3[q3 == np.finfo(np.float32).max] = np.nan

    return q1.astype(np.float32), q3.astype(np.float32)


def compute_iqr_bounds(
    vx_stack: np.ndarray,
    vy_stack: np.ndarray,
    ref_vx: Optional[np.ndarray] = None,
    ref_vy: Optional[np.ndarray] = None,
    T: float = 1.5,
    iqr_max_cap: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute IQR-based acceptance bounds for Vx and Vy stacks.
    Yang et al. (2026): VV = [Q1 - T*IQR,  Q3 + T*IQR]
    """
    def _bounds(stack, ref):
        q1, q3 = _fast_q1q3(stack)
        iqr = q3 - q1

        capped = iqr > iqr_max_cap
        if np.any(capped):
            if ref is None:
                # Fast median via partition at midpoint
                N = stack.shape[0]
                s = stack.copy(); s[~np.isfinite(s)] = np.finfo(np.float32).max
                mid = np.partition(s, N // 2, axis=0)[N // 2]
                mid[mid == np.finfo(np.float32).max] = np.nan
                ref_vals = mid.astype(np.float32)
            else:
                ref_vals = ref
            q1  = np.where(capped, ref_vals - iqr_max_cap / 2, q1)
            q3  = np.where(capped, ref_vals + iqr_max_cap / 2, q3)
            iqr = np.where(capped, iqr_max_cap, iqr)

        lo = (q1 - T * iqr).astype(np.float32)
        hi = (q3 + T * iqr).astype(np.float32)
        return lo, hi

    vx_lo, vx_hi = _bounds(vx_stack, ref_vx)
    vy_lo, vy_hi = _bounds(vy_stack, ref_vy)
    return vx_lo, vx_hi, vy_lo, vy_hi


def apply_temporal_mad_filter(
    vx_stack:     np.ndarray,    # (N, H, W) float32
    vy_stack:     np.ndarray,
    v_stack:      np.ndarray,
    vx_err_stack: np.ndarray,
    vy_err_stack: np.ndarray,
    k:            float = 2.0,
    min_scenes:   int   = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel temporal MAD filter (after Solgaard et al. / Yang et al.).

    For each pixel (i, j), across all N scenes in the group:

        median_vx  = nanmedian( vx[:, i, j] )
        MAD_vx     = nanmedian( |vx[:, i, j] − median_vx| )
        bad_vx     = |vx[n, i, j] − median_vx| > k · MAD_vx

    Same logic applied independently to Vy.  A pixel flagged bad in either
    Vx OR Vy is masked in all bands (Vx, Vy, V, σVx, σVy).

    Parameters
    ----------
    vx_stack, vy_stack, v_stack,
    vx_err_stack, vy_err_stack : (N, H, W) float32 stacks
    k            : MAD multiplier threshold (default 2.0)
    min_scenes   : minimum valid observations per pixel to apply the filter;
                   pixels with fewer scenes are left untouched to avoid
                   removing data where MAD is statistically meaningless.

    Returns
    -------
    Filtered copies of all five stacks (NaN where flagged).
    """
    def _mad_mask(stack: np.ndarray) -> np.ndarray:
        """Return boolean bad-pixel mask (N, H, W)."""
        n_valid = np.sum(np.isfinite(stack), axis=0)          # (H, W)
        median  = np.nanmedian(stack, axis=0)                  # (H, W)
        mad     = np.nanmedian(
            np.abs(stack - median[np.newaxis]), axis=0)        # (H, W)

        # Only filter where we have enough scenes AND MAD is non-zero.
        # MAD == 0 means all values identical → nothing to filter.
        can_filter = (n_valid >= min_scenes) & (mad > 0)

        with np.errstate(invalid="ignore"):
            bad = np.abs(stack - median[np.newaxis]) > k * mad[np.newaxis]

        # Restrict to filterable pixels; leave already-NaN slots alone
        bad &= can_filter[np.newaxis]
        bad[~np.isfinite(stack)] = False
        return bad

    bad_vx  = _mad_mask(vx_stack)
    bad_vy  = _mad_mask(vy_stack)
    bad_any = bad_vx | bad_vy          # propagate to all bands

    vx  = vx_stack.copy();   vx [bad_any] = np.nan
    vy  = vy_stack.copy();   vy [bad_any] = np.nan
    v   = v_stack.copy();    v  [bad_any] = np.nan
    evx = vx_err_stack.copy(); evx[bad_any] = np.nan
    evy = vy_err_stack.copy(); evy[bad_any] = np.nan

    # Stats for logging
    n_finite_in = int(np.sum(np.isfinite(vx_stack)))
    n_removed   = int(np.sum(bad_any & np.isfinite(vx_stack)))
    pct = 100.0 * n_removed / n_finite_in if n_finite_in else 0.0
    log.debug(
        "Temporal MAD filter (k=%.1f, min_n=%d): removed %d / %d "
        "pixel-obs (%.1f%%)",
        k, min_scenes, n_removed, n_finite_in, pct)

    return vx, vy, v, evx, evy


def apply_temporal_mad_filter(
    vx_stack:     np.ndarray,    # (N, H, W) float32
    vy_stack:     np.ndarray,
    v_stack:      np.ndarray,
    vx_err_stack: np.ndarray,
    vy_err_stack: np.ndarray,
    k:            float = 2.0,
    min_scenes:   int   = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel temporal MAD filter applied within one group's scene stack.

    Algorithm (identical for Vx and Vy, applied independently):

        median[i,j] = nanmedian( stack[:, i, j] )
        MAD[i,j]    = nanmedian( |stack[:, i, j] - median[i,j]| )
        bad[n, i,j] = |stack[n, i,j] - median[i,j]| > k * MAD[i,j]

    A pixel flagged bad in Vx OR Vy is masked in ALL bands
    (Vx, Vy, V, σVx, σVy) to keep the bands consistent.

    Guard conditions — the mask is NOT applied when:
      • fewer than `min_scenes` valid observations exist at that pixel
        (MAD from 1-2 points is statistically meaningless)
      • MAD == 0 (all scenes agree exactly — nothing to flag)

    Parameters
    ----------
    k           : MAD multiplier (default 2.0; use 3.0 for looser filter)
    min_scenes  : minimum valid obs per pixel to activate the filter

    Returns
    -------
    Filtered copies of (vx, vy, v, vx_err, vy_err) with NaN where flagged.
    """
    def _mad_mask(stack: np.ndarray) -> np.ndarray:
        """Boolean bad-pixel mask (N, H, W). True = should be NaN-ed."""
        n_valid = np.sum(np.isfinite(stack), axis=0)          # (H, W)
        median  = np.nanmedian(stack, axis=0)                  # (H, W)
        mad     = np.nanmedian(
            np.abs(stack - median[np.newaxis]), axis=0)        # (H, W)

        # Only filter where we have enough scenes AND the spread is non-zero
        can_filter = (n_valid >= min_scenes) & (mad > 0)      # (H, W)

        with np.errstate(invalid="ignore"):
            bad = np.abs(stack - median[np.newaxis]) > k * mad[np.newaxis]

        bad &= can_filter[np.newaxis]        # zero-out un-filterable pixels
        bad[~np.isfinite(stack)] = False     # already-NaN slots are not "bad"
        return bad

    bad_vx  = _mad_mask(vx_stack)           # (N, H, W)
    bad_vy  = _mad_mask(vy_stack)
    bad_any = bad_vx | bad_vy               # propagate to all bands

    vx  = vx_stack.copy();    vx [bad_any] = np.nan
    vy  = vy_stack.copy();    vy [bad_any] = np.nan
    v   = v_stack.copy();     v  [bad_any] = np.nan
    evx = vx_err_stack.copy(); evx[bad_any] = np.nan
    evy = vy_err_stack.copy(); evy[bad_any] = np.nan

    n_finite_in = int(np.sum(np.isfinite(vx_stack)))
    n_removed   = int(np.sum(bad_any & np.isfinite(vx_stack)))
    pct = 100.0 * n_removed / n_finite_in if n_finite_in else 0.0
    log.debug(
        "Temporal MAD filter (k=%.1f, min_n=%d): removed %d / %d "
        "pixel-obs (%.1f%%)", k, min_scenes, n_removed, n_finite_in, pct)

    return vx, vy, v, evx, evy


def apply_iqr_filter(
    vx_stack:    np.ndarray,
    vy_stack:    np.ndarray,
    v_stack:     np.ndarray,
    vx_err_stack: np.ndarray,
    vy_err_stack: np.ndarray,
    ref_vx:      Optional[np.ndarray] = None,
    ref_vy:      Optional[np.ndarray] = None,
    T:           float = 1.5,
    iqr_max_cap: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply IQR filter: set out-of-range pixels to NaN in-place (copy returned).

    Returns filtered copies of (vx_stack, vy_stack, v_stack, vx_err_stack, vy_err_stack).
    """
    vx_lo, vx_hi, vy_lo, vy_hi = compute_iqr_bounds(
        vx_stack, vy_stack, ref_vx, ref_vy, T, iqr_max_cap
    )

    vx  = vx_stack.copy()
    vy  = vy_stack.copy()
    v   = v_stack.copy()
    evx = vx_err_stack.copy()
    evy = vy_err_stack.copy()

    # Mask Vx outliers (and propagate to all bands for that pixel)
    bad_x = (vx < vx_lo[np.newaxis]) | (vx > vx_hi[np.newaxis])
    bad_y = (vy < vy_lo[np.newaxis]) | (vy > vy_hi[np.newaxis])

    vx [bad_x] = np.nan;  evx[bad_x] = np.nan
    vy [bad_y] = np.nan;  evy[bad_y] = np.nan

    # Combined mask on V
    bad_any = bad_x | bad_y
    v[bad_any] = np.nan

    n_removed_x = int(np.sum(bad_x & ~np.isnan(vx_stack)))
    n_removed_y = int(np.sum(bad_y & ~np.isnan(vy_stack)))
    log.debug("IQR filter removed %d Vx, %d Vy outlier pixels",
              n_removed_x, n_removed_y)

    return vx, vy, v, evx, evy
# ─────────────────────────────────────────────────────────────────────────────
# Spatial MAD filter  (paper Stage 1 — single-scene, per-record)
#
# Reference: Remote Sens. 2023, 15, 3079 — Section 2.2 (Spatial Filtering)
#
# Algorithm (applied independently to Vx and Vy):
#   Within a (window_size × window_size) moving window centred on each pixel:
#       μ   = median of valid neighbours
#       σ   = population std of valid neighbours
#       MAD = median absolute deviation
#
#   The centre pixel is flagged erroneous if EITHER criterion fires:
#       (a) |Vx − μ_vx| > MAD_vx  OR  |Vy − μ_vy| > MAD_vy   (outlier)
#       (b)  σ_vx > sigma_threshold  OR  σ_vy > sigma_threshold (high-spread)
#
#   When flagged, the pixel is NaN-ed in ALL provided bands (vx, vy, v, errors).
# ─────────────────────────────────────────────────────────────────────────────

def _window_stats_2d(
    arr: np.ndarray,
    window_size: int,
    min_valid: int = 4,
    tile_rows: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window median, population std, and MAD for a 2D float32 array.

    NaN values inside the window are silently excluded from every statistic.
    Pixels whose window contains fewer than ``min_valid`` finite neighbours
    receive NaN for all three outputs (no false positives at data edges).

    Implementation
    --------------
    Processes the array in horizontal stripes of ``tile_rows`` rows to cap
    peak memory.  Each tile is expanded with ``window_size//2`` padding rows
    (read from the already-padded full array) so window statistics are exact
    at tile boundaries — no stitching artefacts.

    Memory per call ≈ tile_rows × W × ws² × 4 bytes (float32).
    Example: ws=5, tile_rows=256, W=1 850 → ≈ 47 MB per tile — safe for
    production scenes at 120 m resolution over Cook Ice Shelf.

    Parameters
    ----------
    arr         : (H, W) float32 with NaN for missing values
    window_size : odd int side of the square window (default 5)
    min_valid   : minimum finite pixels required in a window
    tile_rows   : rows per processing stripe (tune to available RAM)

    Returns
    -------
    median : (H, W) float32
    std    : (H, W) float32  (population, ddof=0)
    mad    : (H, W) float32
    """
    from numpy.lib.stride_tricks import sliding_window_view

    arr  = arr.astype(np.float32, copy=False)
    H, W = arr.shape
    pad  = window_size // 2

    # Pad once (rows + cols) so tile slices can reach all neighbours cleanly.
    padded = np.pad(arr, pad, mode="constant", constant_values=np.nan)

    median_out = np.full((H, W), np.nan, dtype=np.float32)
    std_out    = np.full((H, W), np.nan, dtype=np.float32)
    mad_out    = np.full((H, W), np.nan, dtype=np.float32)

    for rs in range(0, H, tile_rows):
        re     = min(rs + tile_rows, H)
        tile_H = re - rs

        # padded[rs : re+2p, :] supplies the full window neighbourhood for
        # every output row in [rs, re).  No extra boundary handling needed.
        tile_pad = padded[rs : re + 2 * pad, :]        # (tile_H+2p, W+2p)

        # VIEW  (tile_H, W, ws, ws) — no copy until .reshape().copy()
        wins = sliding_window_view(tile_pad, (window_size, window_size))
        flat = wins.reshape(tile_H, W, -1).copy()      # ← peak allocation

        valid   = np.isfinite(flat)
        n_valid = valid.sum(axis=2).astype(np.int32)   # (tile_H, W)

        # ── Median ────────────────────────────────────────────────────────────
        flat_nan  = np.where(valid, flat, np.nan)
        tile_med  = np.nanmedian(flat_nan, axis=2).astype(np.float32)

        # ── Population Std  (sum-of-squares, single-pass, no extra copy) ──────
        v0   = np.where(valid, flat, 0.0)
        sum1 = v0.sum(axis=2)
        sum2 = (v0 ** 2).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_w = np.where(n_valid > 0, sum1 / n_valid, np.nan)
            var_w  = np.where(
                n_valid > 1,
                np.maximum(sum2 / n_valid - mean_w ** 2, 0.0),
                np.nan,
            )
        tile_std = np.sqrt(var_w).astype(np.float32)

        # ── MAD ───────────────────────────────────────────────────────────────
        abs_dev  = np.where(valid, np.abs(flat_nan - tile_med[:, :, np.newaxis]), np.nan)
        tile_mad = np.nanmedian(abs_dev, axis=2).astype(np.float32)

        # Suppress sparse-window pixels (no false positives at data edges)
        sparse           = n_valid < min_valid
        tile_med[sparse] = np.nan
        tile_std[sparse] = np.nan
        tile_mad[sparse] = np.nan

        median_out[rs:re] = tile_med
        std_out   [rs:re] = tile_std
        mad_out   [rs:re] = tile_mad

        # Free the tile's large temporaries before the next iteration
        del flat, flat_nan, abs_dev, v0

    return median_out, std_out, mad_out


def apply_spatial_mad_filter(
    vx:      np.ndarray,
    vy:      np.ndarray,
    v:       Optional[np.ndarray] = None,
    vx_err:  Optional[np.ndarray] = None,
    vy_err:  Optional[np.ndarray] = None,
    v_err:   Optional[np.ndarray] = None,
    window_size:     int   = 5,
    sigma_threshold: float = 150.0,
    min_valid:       int   = 4,
    tile_rows:       int   = 256,
) -> Tuple[
    np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray],
    dict,
]:
    """
    Single-scene spatial MAD filter — Stage 1 of the paper's spatiotemporal filter.

    For each pixel, statistics are computed over a ``window_size × window_size``
    neighbourhood.  The pixel is removed (set to NaN in all bands) when either
    of two criteria is met:

        (a)  |Vx − μ_vx| > MAD_vx  OR  |Vy − μ_vy| > MAD_vy
             → the pixel deviates from its local median by more than one MAD

        (b)  σ_vx > sigma_threshold  OR  σ_vy > sigma_threshold
             → the whole window is intrinsically high-variance
               (paper threshold: 150 m/yr; chosen to exceed the expected 
                within-2×2-km natural variability of ~100 m/yr on Cook Shelf)

    Pixels at data-sparse edges (fewer than ``min_valid`` finite neighbours)
    are never flagged — both criteria require finite statistics to fire.

    Parameters
    ----------
    vx, vy           : (H, W) float32  velocity components for one scene
    v                : (H, W) float32  speed (optional companion band)
    vx_err, vy_err   : (H, W) float32  velocity errors (optional)
    v_err            : (H, W) float32  speed error (optional)
    window_size      : odd int, side of moving window in pixels (default 5)
    sigma_threshold  : std upper limit in m/yr — paper value 150 m/yr
    min_valid        : min finite pixels in window to compute statistics
    tile_rows        : rows per processing stripe (default 256, tune to RAM)

    Returns
    -------
    vx_f, vy_f       : filtered velocity components
    v_f              : filtered speed (None if not supplied)
    vx_err_f, vy_err_f, v_err_f : filtered errors (None if not supplied)
    stats            : diagnostic dict with removal counts
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    # ── Work on copies so the caller's arrays are never modified ──────────────
    vx_f   = vx.copy()
    vy_f   = vy.copy()
    v_f    = None if v      is None else v.copy()
    vxe_f  = None if vx_err is None else vx_err.copy()
    vye_f  = None if vy_err is None else vy_err.copy()
    ve_f   = None if v_err  is None else v_err.copy()

    # ── Compute sliding-window statistics for each component ──────────────────
    mu_vx, sigma_vx, mad_vx = _window_stats_2d(vx, window_size, min_valid, tile_rows)
    mu_vy, sigma_vy, mad_vy = _window_stats_2d(vy, window_size, min_valid, tile_rows)

    # ── Criterion (a): centre deviates from local median by > 1 MAD ──────────
    with np.errstate(invalid="ignore"):
        crit_a = (
            (np.abs(vx - mu_vx) > mad_vx) |
            (np.abs(vy - mu_vy) > mad_vy)
        )

    # ── Criterion (b): window has too much spread (bad acquisition area) ──────
    with np.errstate(invalid="ignore"):
        crit_b = (sigma_vx > sigma_threshold) | (sigma_vy > sigma_threshold)

    # Only flag pixels that are themselves finite (don't re-count existing NaNs)
    is_valid = np.isfinite(vx) & np.isfinite(vy)
    bad      = (crit_a | crit_b) & is_valid

    # ── Apply mask to all bands ───────────────────────────────────────────────
    for arr in (vx_f, vy_f, v_f, vxe_f, vye_f, ve_f):
        if arr is not None:
            arr[bad] = np.nan

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n_in   = int(is_valid.sum())
    n_out  = int(bad.sum())
    pct    = 100.0 * n_out / n_in if n_in else 0.0

    # Count exclusive contributions of each criterion (b adds on top of a)
    n_a    = int((crit_a & is_valid).sum())
    n_b    = int((crit_b & ~crit_a & is_valid).sum())   # b-only pixels

    stats = dict(
        n_input_valid = n_in,
        n_removed     = n_out,
        removal_pct   = round(pct, 2),
        n_crit_a      = n_a,      # outlier criterion
        n_crit_b_only = n_b,      # high-variance criterion (not already caught by a)
    )

    log.debug(
        "SpatialMAD (ws=%d, σ_thr=%.0f): removed %d / %d px "
        "(%.1f%%)  [crit_a=%d  crit_b_only=%d]",
        window_size, sigma_threshold,
        n_out, n_in, pct, n_a, n_b,
    )

    return vx_f, vy_f, v_f, vxe_f, vye_f, ve_f, stats
