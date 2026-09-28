"""
weighted_avg.py
===============
Inverse-variance weighted average for Vx, Vy, V stacks.
V is also synthesised from Vx/Vy with proper error propagation.

Weights  :  w_i = 1 / σ_i²
Mean     :  v̄ = Σ(w_i · v_i) / Σ(w_i)
Error    :  σ_v̄ = 1 / √(Σ(w_i))
n_eff    :  (Σw_i)² / Σ(w_i²)   [effective degrees of freedom]

V synthesis:
  V     = √(Vx² + Vy²)
  σ_V   = √( (Vx/V · σVx)² + (Vy/V · σVy)² )
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)


def weighted_average(
    values:  np.ndarray,          # (N, H, W) float32
    errors:  np.ndarray,          # (N, H, W) float32  1-sigma
    min_obs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform inverse-variance weighted average along axis-0.

    Returns
    -------
    mean  : (H, W) float32   weighted mean   – NaN where obs < min_obs
    err   : (H, W) float32   propagated σ    – NaN where obs < min_obs
    n_eff : (H, W) float32   effective count – NaN where obs < min_obs
    """
    N, H, W = values.shape

    # Build weight mask: both value and error must be finite and positive
    pos_err = np.abs(errors)
    pos_err[pos_err < 1e-10] = np.nan

    valid = np.isfinite(values) & np.isfinite(pos_err)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(valid, 1.0 / (pos_err ** 2), 0.0)

    sum_w  = np.sum(w, axis=0)                             # (H, W)
    sum_wv = np.sum(np.where(valid, w * values, 0.0), axis=0)  # (H, W)
    sum_w2 = np.sum(w ** 2, axis=0)                        # (H, W)
    n_obs  = np.sum(valid, axis=0).astype(np.int32)        # (H, W)

    enough = (n_obs >= min_obs) & (sum_w > 0)

    mean  = np.full((H, W), np.nan, dtype=np.float32)
    err   = np.full((H, W), np.nan, dtype=np.float32)
    n_eff = np.full((H, W), np.nan, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean[enough]  = (sum_wv[enough] / sum_w[enough]).astype(np.float32)
        err [enough]  = (1.0 / np.sqrt(sum_w[enough])).astype(np.float32)
        n_eff[enough] = (sum_w[enough]**2 / sum_w2[enough]).astype(np.float32)

    return mean, err, n_eff


def synthesise_v(
    vx:    np.ndarray,   # (H, W)
    vy:    np.ndarray,   # (H, W)
    vx_err: np.ndarray,  # (H, W)
    vy_err: np.ndarray,  # (H, W)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute V = √(Vx² + Vy²) and propagate errors.

    σ_V = √( (Vx/V · σVx)² + (Vy/V · σVy)² )

    Returns
    -------
    v     : (H, W) float32
    v_err : (H, W) float32
    """
    v = np.sqrt(vx**2 + vy**2).astype(np.float32)

    safe_v = np.where(v > 1e-8, v, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        v_err = np.sqrt(
            (vx / safe_v * vx_err)**2 +
            (vy / safe_v * vy_err)**2
        ).astype(np.float32)

    v_err[np.isnan(safe_v)] = np.nan

    return v, v_err


def process_stack(
    vx_stack:     np.ndarray,   # (N, H, W)
    vy_stack:     np.ndarray,
    v_stack:      np.ndarray,   # direct V from offset tracking (may differ from synthesis)
    vx_err_stack: np.ndarray,
    vy_err_stack: np.ndarray,
    v_err_stack:  np.ndarray,   # (N, H, W) – may be NaN-filled if derived
    min_obs:      int = 1,
) -> dict:
    """
    Full weighted-average pass for all three components plus synthesised V.

    Returns a dict with keys:
      'vx', 'vx_err', 'neff_vx',
      'vy', 'vy_err', 'neff_vy',
      'v',  'v_err',  'neff_v',
      'v_synth', 'v_synth_err'   ← V from error propagation of Vx/Vy
    """
    N = vx_stack.shape[0]

    if N == 1:
        # Single image – pass through, compute neff trivially
        def _neff1(arr):
            n = np.where(np.isfinite(arr[0]), 1.0, np.nan).astype(np.float32)
            return n

        vx_m, vx_e = vx_stack[0], vx_err_stack[0]
        vy_m, vy_e = vy_stack[0], vy_err_stack[0]
        v_m,  v_e  = v_stack[0],  v_err_stack[0]

        result = dict(
            vx=vx_m, vx_err=vx_e, neff_vx=_neff1(vx_stack),
            vy=vy_m, vy_err=vy_e, neff_vy=_neff1(vy_stack),
            v =v_m,  v_err =v_e,  neff_v =_neff1(v_stack),
        )
    else:
        vx_m, vx_e, neff_vx = weighted_average(vx_stack, vx_err_stack, min_obs)
        vy_m, vy_e, neff_vy = weighted_average(vy_stack, vy_err_stack, min_obs)
        v_m,  v_e,  neff_v  = weighted_average(v_stack,  v_err_stack,  min_obs)

        result = dict(
            vx=vx_m, vx_err=vx_e, neff_vx=neff_vx,
            vy=vy_m, vy_err=vy_e, neff_vy=neff_vy,
            v =v_m,  v_err =v_e,  neff_v =neff_v,
        )

    # Synthesised V (error propagation from Vx / Vy)
    v_synth, v_synth_err = synthesise_v(
        result["vx"], result["vy"],
        result["vx_err"], result["vy_err"],
    )
    result["v_synth"]     = v_synth
    result["v_synth_err"] = v_synth_err

    return result
