"""Read assimilation results back from the experiment's HDF5 file.

The dataset names are the ones written by ``src/``:

* standard method (``analysis_nDrif.multi_analysis_optm``): names end with
  ``_{n}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}``.
  ``src/`` builds them inline, so ``std_suffix`` rebuilds that format here.
* sliding method (``sliding_window_assim.multi_analysis_sliding_window``):
  names come from ``_make_suffix`` / ``_win_name``, imported from ``src/``.

Every array is returned as NumPy. Trajectories are returned with the window's
start position prepended, so index ``j`` is the position at
``window start + j * dt_advection``.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .config import Config

METHODS = ("standard", "sliding")


# --------------------------------------------------------------------------- #
# Dataset names
# --------------------------------------------------------------------------- #
def std_suffix(cfg: Config) -> str:
    return (f"_{cfg.n_days}days_fil{cfg.lfil}_alp{cfg.alpha:.6f}_bet{cfg.beta:.6f}"
            f"_toll{cfg.toll:.2f}_lr{cfg.mylr:.6f}")


def sld_suffix(cfg: Config) -> str:
    from sliding_window_assim import _make_suffix
    return _make_suffix(cfg.n_days, cfg.lfil, cfg.alpha, cfg.beta, cfg.toll,
                        cfg.mylr, cfg.tw_days, cfg.sigma_days)


def _suffix(cfg: Config, method: str) -> str:
    return std_suffix(cfg) if method == "standard" else sld_suffix(cfg)


def n_windows(h5, method: str, cfg: Config) -> int:
    if method == "standard":
        return cfg.n_corr
    pattern = re.compile(r"dU_win001of(\d{3})" + re.escape(sld_suffix(cfg)) + "$")
    for name in h5[method]["dU"]:
        m = pattern.match(name)
        if m:
            return int(m.group(1))
    raise KeyError("No sliding-window corrections found for this configuration.")


def window_spans(cfg: Config, method: str, n_win: int) -> list[tuple[float, float]]:
    """(start_day, end_day) of each correction window, relative to the run start."""
    if method == "standard":
        d = cfg.n_days // cfg.n_corr
        return [(w * d, (w + 1) * d) for w in range(n_win)]
    return [(k * cfg.sigma_days, k * cfg.sigma_days + cfg.tw_days) for k in range(n_win)]


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
def has_results(h5, method: str, cfg: Config) -> bool:
    """True when this method already ran with exactly this configuration."""
    if method not in h5:
        return False
    grp = h5[method]
    key = f"errors/Relative_Delta_RMS{_suffix(cfg, method)}"
    return key in grp and grp.attrs.get("fingerprint") == cfg.fingerprint()


def mark_results(h5, method: str, cfg: Config, runtime_s: float) -> None:
    grp = h5[method]
    grp.attrs["fingerprint"] = cfg.fingerprint()
    grp.attrs["runtime_s"] = float(runtime_s)
    grp.attrs["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")


def save_fields(h5, method: str, **arrays) -> None:
    """Store arrays that ``src/`` returns but does not save (sliding ``Uc, Vc``)."""
    grp = h5[method].require_group("notebook")
    for name, arr in arrays.items():
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=to_numpy(arr), compression="gzip")


def load_fields(h5, method: str, *names):
    grp = h5[method]["notebook"]
    return tuple(grp[n][:] for n in names)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mask_track(lon, lat):
    """``simul_drifter`` leaves (0, 0) after a drifter exits the domain."""
    lon = np.array(lon, dtype=float)
    lat = np.array(lat, dtype=float)
    out = (lon == 0) & (lat == 0)
    lon[out] = np.nan
    lat[out] = np.nan
    return lon, lat


def load_rms(h5, method: str, cfg: Config) -> dict:
    """Relative field RMSE (%) of background and analysis at every advection step."""
    sfx = _suffix(cfg, method)
    err = h5[method]["errors"]
    back = err[f"RMS_back{sfx}"][:]
    corr = err[f"RMS_corr{sfx}"][:]
    return {
        "t_days": np.arange(back.size) * cfg.dt_advection / 86400.0,
        "back": back,
        "corr": corr,
        "drms": 100.0 * (back - corr) / back,   # reduction, % of the background error
        "rel": float(err[f"Relative_Delta_RMS{sfx}"][()]),
    }


def load_objective(h5, method: str, cfg: Config) -> list[np.ndarray]:
    """Objective-function value at each Adam iteration, one array per window."""
    grp = h5[method]["objective_function"]
    K = n_windows(h5, method, cfg)
    if method == "standard":
        sfx = std_suffix(cfg)
        return [grp[f"obj_fct_behavior_window{w + 1}{sfx}"][:] for w in range(K)]
    sfx = sld_suffix(cfg)
    return [grp[f"obj_fct_win{k + 1:03d}of{K:03d}{sfx}"][:] for k in range(K)]


def load_dU(h5, method: str, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unfiltered correction (dU, dV) of each window, shape (nlon, nlat)."""
    grp = h5[method]["dU"]
    K = n_windows(h5, method, cfg)
    out = []
    for w in range(K):
        if method == "standard":
            sfx = std_suffix(cfg)
            du = grp[f"dUincrement_{w + 1}window{sfx}"][:]
            dv = grp[f"dVincrement_{w + 1}window{sfx}"][:]
        else:
            sfx = sld_suffix(cfg)
            du = grp[f"dU_win{w + 1:03d}of{K:03d}{sfx}"][:]
            dv = grp[f"dV_win{w + 1:03d}of{K:03d}{sfx}"][:]
        out.append((np.squeeze(du), np.squeeze(dv)))
    return out


def load_obs(h5, cfg: Config):
    """Observed (synthetic) tracks, shape (n_drif, n_days * steps_per_day + 1).

    Column 0 is the release point.
    """
    grp = h5["metadata"]
    lons, lats = [], []
    for nd in range(1, cfg.n_drif + 1):
        lo = grp[f"lon_obs0{nd}_{cfg.n_days}days"][:].reshape(-1)
        la = grp[f"lat_obs0{nd}_{cfg.n_days}days"][:].reshape(-1)
        lo, la = mask_track(lo, la)
        lons.append(np.concatenate([[cfg.lon0[nd - 1]], lo]))
        lats.append(np.concatenate([[cfg.lat0[nd - 1]], la]))
    return np.array(lons), np.array(lats)


def load_window_traj(h5, method: str, cfg: Config, kind: str):
    """Per-window trajectories, ``kind`` = "back" (dU = 0) or "last" (analysis).

    Returns a list over windows of (lon, lat) arrays of shape (n_drif, steps + 1).
    The prepended start point is the one ``src/`` used: the release point for
    the first window, the observation at the window start afterwards.
    """
    obs_lon, obs_lat = load_obs(h5, cfg)
    K = n_windows(h5, method, cfg)
    spans = window_spans(cfg, method, K)
    grp = h5[method][f"traj_{kind}"]
    out = []
    for w in range(K):
        start = int(spans[w][0] * cfg.steps_per_day)
        # src takes obs[start] (no prepended release point) except at t = 0
        s_idx = 0 if start == 0 else start + 1
        lons, lats = [], []
        for nd in range(1, cfg.n_drif + 1):
            if method == "standard":
                sfx = std_suffix(cfg)
                lo = grp[f"lon_traj0{nd}_{kind}{w + 1}window{sfx}"][:]
                la = grp[f"lat_traj0{nd}_{kind}{w + 1}window{sfx}"][:]
            else:
                from sliding_window_assim import _win_name
                args = (nd, w, K, cfg.n_days, cfg.lfil, cfg.alpha, cfg.beta,
                        cfg.toll, cfg.mylr, cfg.tw_days, cfg.sigma_days)
                lo = grp[_win_name(f"lon_traj_{kind}", *args)][:]
                la = grp[_win_name(f"lat_traj_{kind}", *args)][:]
            lo, la = mask_track(lo.reshape(-1), la.reshape(-1))
            lons.append(np.concatenate([[obs_lon[nd - 1, s_idx]], lo]))
            lats.append(np.concatenate([[obs_lat[nd - 1, s_idx]], la]))
        out.append((np.array(lons), np.array(lats)))
    return out


def iterations(h5, method: str, cfg: Config) -> list[int]:
    return [len(o) for o in load_objective(h5, method, cfg)]


# --------------------------------------------------------------------------- #
# Extra trajectories for the movies (thin wrappers around src/simul_drifter)
# --------------------------------------------------------------------------- #
def advect_all(cfg: Config, U, V, lon, lat):
    """Advect every release point through daily fields ``U, V`` (time, lon, lat).

    Returns NumPy (lon, lat) of shape (n_drif, (len(U) - 1) * steps_per_day + 1),
    release point first.
    """
    import torch
    from adv_torch import simul_drifter

    lons, lats = [], []
    for lo0, la0 in zip(cfg.lon0, cfg.lat0):
        x0 = torch.tensor([lo0], device=U.device, dtype=U.dtype)
        y0 = torch.tensor([la0], device=U.device, dtype=U.dtype)
        with torch.no_grad():
            lo, la = simul_drifter(mylonIN=x0, mylatIN=y0, Ut=U, Vt=V,
                                   lon=lon, lat=lat, dtAdvection=cfg.dt_advection)
        lo, la = mask_track(to_numpy(lo).reshape(-1), to_numpy(la).reshape(-1))
        lons.append(np.concatenate([[lo0], lo]))
        lats.append(np.concatenate([[la0], la]))
    return np.array(lons), np.array(lats)


def daily_from_steps(F, steps_per_day: int):
    """Daily snapshots of a field given at every advection step.

    ``multi_analysis_sliding_window`` returns ``n_days * steps_per_day`` steps
    (the final instant is dropped), so the last day reuses the last step.
    """
    n_days = F.shape[0] // steps_per_day
    idx = [d * steps_per_day for d in range(n_days)] + [F.shape[0] - 1]
    return F[idx]
