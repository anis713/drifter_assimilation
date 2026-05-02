"""
sliding_window_assim.py
=======================
Sliding-window variational assimilation of surface drifter data.

Implements Eq. 5 of Issa et al. (2016), Ocean Modelling 104, 1-14:

    u_corrected(t_i) = u_b(t_i) + sum_{k in W(t_i)} w_k * delta_u_k

where  w_k = 1 / (|k - k*| + 1),  normalised to sum to 1,
       k*  = window whose centre is closest to t_i,
       W(t_i) = all windows whose time interval contains t_i.

Two-phase algorithm
-------------------
Phase 1: For every sliding window k, minimise the cost function (Eq. 4)
         to obtain a time-independent spatial correction delta_u_k.
Phase 2: Reconstruct the time-continuous corrected field by applying
         the weighted superposition (Eq. 5).

Dependencies (must be importable from the caller's environment):
    GenerateCoefs, ApplyFilter, field_int_new, simul_drifter,
    diverg_penalty, RMS_per_window_new, check_if_exists
"""

import numpy as np
import torch
import torch.nn.functional as F
from decimal import getcontext
from analysis_nDrif import field_int_new, check_if_exists, RMS_per_window_new
from adv_torch import GenerateCoefs, simul_drifter, ApplyFilter, diverg_penalty


# ======================================================================== #
# Helper utilities
# ======================================================================== #

def _make_suffix(nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days):
    """Shared parameter string appended to every HDF5 dataset name."""
    return (
        f"_Tw{Tw_days}d_sig{sigma_days}d_{nDaysSim}days_fil{lfil}"
        f"_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"
    )


def _win_name(prefix, nd, k, K, nDaysSim, lfil,
              alpha, beta, toll, mylr, Tw_days, sigma_days):
    """Dataset name for a per-window, per-drifter quantity."""
    suffix = _make_suffix(nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days)
    return f"{prefix}_nd{nd:02d}_win{k+1:03d}of{K:03d}{suffix}"


# ======================================================================== #
# Main function
# ======================================================================== #

def multi_analysis_sliding_window(
    Ut, Vt, Ub, Vb, lon, lat, lon0, lat0,
    maskuu, maskvv, assim_grp, obs_grp,
    nDaysSim, lfil, dtAdvection, dtObservation,
    mylr, alpha, beta,
    n_drif, Tw_days, sigma_days,
    toll, synth_traj=False, device=None
):
    """
    Sliding-window variational assimilation — Eq. 5 of Issa et al. (2016).

    Parameters
    ----------
    Ut, Vt : torch.Tensor, shape (nDaysSim+1, nlat, nlon)
        "True" velocity field used only for RMS validation.
    Ub, Vb : torch.Tensor, shape (nDaysSim+1, nlat, nlon)
        Background (geostrophic + wind) velocity field (daily sampling).
    lon, lat : 1-D array-like
        Longitude / latitude grid vectors.
    lon0, lat0 : array-like
        Override initial drifter positions (used only when synth_traj=True
        and k=0).
    maskuu, maskvv : array-like
        Land masks for u- and v-velocity components.
    assim_grp : h5py.Group
        Root HDF5 group for assimilation outputs.
    obs_grp : h5py.Group
        HDF5 group containing observed drifter positions.
        Expected dataset keys:
            'lon_obs0{nd}_{nDaysSim}days'  shape (nDaysSim, nObsStepPerDay)
            'lat_obs0{nd}_{nDaysSim}days'  shape (nDaysSim, nObsStepPerDay)
    nDaysSim : int
        Total simulation length in days.
    lfil : int or float
        Spatial filter length scale (passed to GenerateCoefs / ApplyFilter).
    dtAdvection : float
        Advection time step [s].
    dtObservation : float
        Observation sampling interval [s].  Must satisfy dtObservation >= dtAdvection
        and dtObservation % dtAdvection == 0.
    mylr : float
        Adam learning rate.
    alpha : float
        Background-penalty weight (Eq. 4).
    beta : float
        Divergence-penalty weight (Eq. 4).
    n_drif : int
        Number of drifters.
    Tw_days : int
        Assimilation window size [days].  Must be a positive integer.
    sigma_days : float
        Window shift [days].
        Constraint: sigma_days * 24 * 3600 / dtObservation must be an integer.
        Example: sigma_days=0.25 with dtObservation=21600 s (6 h) → 1 obs step.
    toll : float
        Stopping tolerance on relative gradient-norm change [%].
    synth_traj : bool
        If True, override drifter initial positions for window k=0 with lon0/lat0.
    device : torch.device or None
        Compute device.

    Returns
    -------
    relative_dRMS : float
        Mean relative RMS improvement over the full run [%].
    Uc_full_int : torch.Tensor, shape (nAdvTotal+1, nlat, nlon)
        Corrected u-velocity at advection time resolution.
    Vc_full_int : torch.Tensor, shape (nAdvTotal+1, nlat, nlon)
        Corrected v-velocity at advection time resolution.

    Notes
    -----
    * simul_drifter is called with the *daily* background slice for each
      window (Ub_k_daily), consistent with the existing codebase.  The
      function is expected to interpolate internally from daily to
      dtAdvection resolution.  The correction dU (shape 1×nlat×nlon)
      broadcasts over the time dimension.

    * When sigma_days is not a multiple of 1 day, the daily background
      slice for window k spans floor(k*sigma_days) … ceil(k*sigma_days+Tw_days)
      days.  simul_drifter will therefore simulate slightly more than
      Tw_days of trajectory; only the first Tw_obs_steps observation
      indices are used in the cost function.
    """
    torch.set_printoptions(precision=8)
    getcontext().prec = 8
    dtype = torch.float32

    # ------------------------------------------------------------------ #
    # 0.  Time-step bookkeeping
    # ------------------------------------------------------------------ #
    nObsStepPerDay  = int(round(24.0 * 3600.0 / dtObservation))
    nAdvStepPerDay  = int(round(24.0 * 3600.0 / dtAdvection))
    stride          = int(round(dtObservation / dtAdvection))   # adv steps per obs step

    Tw_obs_steps    = Tw_days * nObsStepPerDay          # window size  [obs steps]
    Tw_adv_steps    = Tw_days * nAdvStepPerDay          # window size  [adv steps]
    sigma_obs_steps = int(round(sigma_days * nObsStepPerDay))   # shift [obs steps]
    sigma_adv_steps = int(round(sigma_days * nAdvStepPerDay))   # shift [adv steps]

    nObsTotal = nDaysSim * nObsStepPerDay   # total obs steps
    nAdvTotal = nDaysSim * nAdvStepPerDay   # total adv steps

    # Total number of sliding windows (Eq. 4 is applied K times)
    K = (nObsTotal - Tw_obs_steps) // sigma_obs_steps + 1

    Nlat = lat.shape
    Nlon = lon.shape

    print(
        f"\n[sliding_window] Tw={Tw_days} d | sigma={sigma_days} d | "
        f"K={K} windows | nObsStepPerDay={nObsStepPerDay} | stride={stride}"
    )

    # ------------------------------------------------------------------ #
    # 1.  Filter / covariance coefficients (shared across all windows)
    # ------------------------------------------------------------------ #
    fileu, filev, dx, dy, masku, maskv, coeffc, coeffd = GenerateCoefs(
        lat, lon, Nlat[0], Nlon[0], maskuu, maskvv, lfil,
        device=device, dtype=dtype
    )

    # ------------------------------------------------------------------ #
    # 2.  Full-run background & truth at advection resolution
    #     Used in Phase 2 for the final corrected field and for RMS.
    #     Ub_full_int shape: (nAdvTotal+1, nlat, nlon)
    # ------------------------------------------------------------------ #
    Ub_daily = Ub.detach()   # (nDaysSim+1, nlat, nlon) – for simul_drifter
    Vb_daily = Vb.detach()
    Ut_daily = Ut.detach()
    Vt_daily = Vt.detach()

    Ub_full_int, Vb_full_int = field_int_new(Ub_daily, Vb_daily, dtAdvection)
    Ut_full_int, Vt_full_int = field_int_new(Ut_daily, Vt_daily, dtAdvection)

    # ------------------------------------------------------------------ #
    # 3.  Pre-load all drifter observations as flat arrays
    #     obs_lon_all shape: (n_drif, nObsTotal)
    # ------------------------------------------------------------------ #
    obs_lon_all = torch.zeros((n_drif, nObsTotal), device=device, dtype=dtype)
    obs_lat_all = torch.zeros((n_drif, nObsTotal), device=device, dtype=dtype)

    for nd in range(1, n_drif + 1):
        lon_arr = obs_grp[f'lon_obs0{nd}_{nDaysSim}days'][:]   # (nDaysSim, nObsStepPerDay)
        lat_arr = obs_grp[f'lat_obs0{nd}_{nDaysSim}days'][:]
        obs_lon_all[nd - 1] = (
            torch.from_numpy(lon_arr).float().to(device).reshape(-1)
        )
        obs_lat_all[nd - 1] = (
            torch.from_numpy(lat_arr).float().to(device).reshape(-1)
        )

    # ------------------------------------------------------------------ #
    # 4.  HDF5 output groups
    # ------------------------------------------------------------------ #
    dU_grp        = assim_grp.require_group("dU")
    obj_fct_grp   = assim_grp.require_group("objective_function")
    err_grp       = assim_grp.require_group("errors")
    traj_back_grp = assim_grp.require_group("traj_back")
    traj_last_grp = assim_grp.require_group("traj_last")

    suffix = _make_suffix(nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days)

    # Obs-step indices within a window used for trajectory downsampling.
    # obs step j  →  adv step j*stride   (starting at t=0, i.e. the
    # initial position is obs step 0).
    sample_idx = torch.arange(Tw_obs_steps, device=device) * stride  # (Tw_obs_steps,)

    # ================================================================== #
    # PHASE 1 — Compute δu_k for every sliding window k
    # ================================================================== #
    # delta_u_list[k] = (dU_k, dV_k) both on CPU, shape (1, nlat, nlon)
    delta_u_list = []

    for k in range(K):
        print(f"\n{'='*64}")
        print(f"  PHASE 1  |  Window {k + 1}/{K}")
        print(f"{'='*64}")

        # ---- Observation index range [obs_start, obs_end) ----------------
        obs_start = k * sigma_obs_steps
        obs_end   = obs_start + Tw_obs_steps   # exclusive (Python slice)

        # ---- Daily background slice for this window ----------------------
        # Window spans obs steps [obs_start, obs_end) which in fractional
        # days is [day_start_f, day_end_f).
        day_start_f = obs_start / nObsStepPerDay
        day_end_f   = obs_end   / nObsStepPerDay

        day_start = int(np.floor(day_start_f))
        day_end   = min(int(np.ceil(day_end_f)), nDaysSim)

        # Shape: (day_end - day_start + 1, nlat, nlon)
        Ub_k_daily = Ub_daily[day_start : day_end + 1]
        Vb_k_daily = Vb_daily[day_start : day_end + 1]

        # ---- Initial drifter positions (first obs of window) -------------
        lon_0 = obs_lon_all[:, obs_start].clone()   # (n_drif,)
        lat_0 = obs_lat_all[:, obs_start].clone()

        if synth_traj and k == 0:
            lon0_t = torch.as_tensor(lon0, device=device, dtype=dtype).flatten()
            lat0_t = torch.as_tensor(lat0, device=device, dtype=dtype).flatten()
            lon_0.copy_(lon0_t)
            lat_0.copy_(lat0_t)

        # ---- Observations for this window (n_drif, Tw_obs_steps) --------
        obs_lon_k = obs_lon_all[:, obs_start:obs_end]  # (n_drif, Tw_obs_steps)
        obs_lat_k = obs_lat_all[:, obs_start:obs_end]

        # ---- Optimisable correction: time-independent within window ------
        dU = torch.nn.Parameter(
            torch.zeros((1, Nlat[0], Nlon[0]), device=device, dtype=dtype)
        )
        dV = torch.nn.Parameter(
            torch.zeros((1, Nlat[0], Nlon[0]), device=device, dtype=dtype)
        )
        optimizer = torch.optim.Adam([dU, dV], lr=mylr)

        # ---- Save background trajectory (delta_u = 0) before optimisation
        with torch.no_grad():
            for nd in range(1, n_drif + 1):
                lon_bk_ds = _win_name(
                    "lon_traj_back", nd, k, K,
                    nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days
                )
                lat_bk_ds = _win_name(
                    "lat_traj_back", nd, k, K,
                    nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days
                )
                check_if_exists(lon_bk_ds, traj_back_grp)
                check_if_exists(lat_bk_ds, traj_back_grp)

                lon_back, lat_back = simul_drifter(
                    mylonIN=lon_0[nd - 1].reshape(1),
                    mylatIN=lat_0[nd - 1].reshape(1),
                    Ut=Ub_k_daily + dU,   # dU=0 → pure background
                    Vt=Vb_k_daily + dV,
                    lon=lon, lat=lat,
                    dtAdvection=dtAdvection,
                )
                traj_back_grp.create_dataset(
                    lon_bk_ds,
                    data=lon_back.detach().cpu().numpy(),
                    compression="gzip",
                )
                traj_back_grp.create_dataset(
                    lat_bk_ds,
                    data=lat_back.detach().cpu().numpy(),
                    compression="gzip",
                )

        # ---- Optimization loop -------------------------------------------
        count            = 0
        diff_norm_perc   = 100.0
        full_grad_norm   = torch.tensor(4.0)
        full_grad_old    = torch.tensor(2.0)
        itr_max          = 25
        patience         = 5
        eps_loss         = 1e-4
        no_improve_count = 0
        best_loss        = float("inf")
        losstotal_values = []

        while diff_norm_perc > toll:
            full_grad_old = full_grad_norm
            optimizer.zero_grad(set_to_none=True)

            # Apply spatial filter (diffusion-based covariance, Weaver & Courtier 2001)
            dUFil, dVFil = ApplyFilter(
                dU, dV, Nlat[0], Nlon[0],
                fileu, filev, dx, dy,
                masku, maskv, coeffc, coeffd, lfil,
            )

            loss_drif = torch.zeros((), device=device, dtype=dtype)

            for nd in range(1, n_drif + 1):
                # Forward model: advect drifter with background + correction
                lon_sim, lat_sim = simul_drifter(
                    mylonIN=lon_0[nd - 1].reshape(1),
                    mylatIN=lat_0[nd - 1].reshape(1),
                    Ut=Ub_k_daily + dUFil,
                    Vt=Vb_k_daily + dVFil,
                    lon=lon, lat=lat,
                    dtAdvection=dtAdvection,
                )

                # Downsample advection trajectory to observation timestep
                # lon_sim shape: (1, nAdvSteps+1)
                # sample_idx selects obs-aligned adv steps → (1, Tw_obs_steps)
                if dtAdvection != dtObservation:
                    lon_adv = lon_sim[:, sample_idx].reshape(-1)
                    lat_adv = lat_sim[:, sample_idx].reshape(-1)
                else:
                    lon_adv = lon_sim[:, :Tw_obs_steps].reshape(-1)
                    lat_adv = lat_sim[:, :Tw_obs_steps].reshape(-1)

                # Observed positions for drifter nd in this window
                obs_lon_nd = obs_lon_k[nd - 1].reshape(-1)  # (1, Tw_obs_steps)
                obs_lat_nd = obs_lat_k[nd - 1].reshape(-1)

                # MSE misfit (Eq. 4 – first term)
                y    = torch.stack([lon_adv, lat_adv])
                yobs = torch.stack([obs_lon_nd, obs_lat_nd])
                loss_drif = loss_drif + F.mse_loss(y, yobs)

            # Regularisation terms (Eq. 4 – second and third terms)
            bg_pen  = torch.norm(dU) + torch.norm(dV)         # background penalty
            div_pen = diverg_penalty(dU, dV, lat, lon)         # divergence penalty

            losstotal = loss_drif + alpha * bg_pen + beta * div_pen
            losstotal.backward()
            optimizer.step()

            current_loss = losstotal.item()
            losstotal_values.append(current_loss)

            # Early-stopping bookkeeping
            if best_loss - current_loss > eps_loss:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Relative gradient-norm change (convergence criterion)
            full_grad_norm = torch.sqrt(
                torch.linalg.norm(dU.grad) ** 2 +
                torch.linalg.norm(dV.grad) ** 2
            )
            diff_norm_perc = float(
                100.0 * abs(full_grad_norm - full_grad_old) /
                (abs(full_grad_old) + 1e-12)
            )

            count += 1
            print(
                f"  iter {count:3d} | loss={current_loss:.6e} | "
                f"grad_diff={diff_norm_perc:.4f}%"
            )

            if count > itr_max and no_improve_count >= patience:
                print("  → Early stopping (loss plateau).")
                break

        # ---- Store δu_k in memory and HDF5 ----------------------------------
        dU_k_cpu = dU.detach().cpu()
        dV_k_cpu = dV.detach().cpu()
        delta_u_list.append((dU_k_cpu, dV_k_cpu))

        du_ds = f"dU_win{k + 1:03d}of{K:03d}{suffix}"
        dv_ds = f"dV_win{k + 1:03d}of{K:03d}{suffix}"
        check_if_exists(du_ds, dU_grp)
        check_if_exists(dv_ds, dU_grp)
        dU_grp.create_dataset(du_ds, data=dU_k_cpu.numpy(), compression="gzip")
        dU_grp.create_dataset(dv_ds, data=dV_k_cpu.numpy(), compression="gzip")

        obj_ds = f"obj_fct_win{k + 1:03d}of{K:03d}{suffix}"
        check_if_exists(obj_ds, obj_fct_grp)
        obj_fct_grp.create_dataset(
            obj_ds, data=np.array(losstotal_values), compression="gzip"
        )

        # ---- Post-optimisation trajectory (with filtered correction) --------
        dUFil_fin, dVFil_fin = ApplyFilter(
            dU, dV, Nlat[0], Nlon[0],
            fileu, filev, dx, dy,
            masku, maskv, coeffc, coeffd, lfil,
        )
        with torch.no_grad():
            for nd in range(1, n_drif + 1):
                lon_last_ds = _win_name(
                    "lon_traj_last", nd, k, K,
                    nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days
                )
                lat_last_ds = _win_name(
                    "lat_traj_last", nd, k, K,
                    nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days
                )
                check_if_exists(lon_last_ds, traj_last_grp)
                check_if_exists(lat_last_ds, traj_last_grp)

                lon_last, lat_last = simul_drifter(
                    mylonIN=lon_0[nd - 1].reshape(1),
                    mylatIN=lat_0[nd - 1].reshape(1),
                    Ut=Ub_k_daily + dUFil_fin,
                    Vt=Vb_k_daily + dVFil_fin,
                    lon=lon, lat=lat,
                    dtAdvection=dtAdvection,
                )
                traj_last_grp.create_dataset(
                    lon_last_ds,
                    data=lon_last.detach().cpu().numpy(),
                    compression="gzip",
                )
                traj_last_grp.create_dataset(
                    lat_last_ds,
                    data=lat_last.detach().cpu().numpy(),
                    compression="gzip",
                )

        print(f"  Window {k + 1}/{K} complete.")

    # ================================================================== #
    # PHASE 2 — Reconstruct the corrected field via Eq. 5
    # ================================================================== #
    print(f"\n{'='*64}")
    print("  PHASE 2  |  Reconstructing corrected field  (Eq. 5)")
    print(f"{'='*64}")

    # Precompute the advection-step interval [start, end] for every window.
    # Window k spans obs steps [k*sigma_obs, k*sigma_obs + Tw_obs),
    # which in adv steps is [k*sigma_adv, k*sigma_adv + Tw_adv].
    win_adv_start = [k * sigma_adv_steps          for k in range(K)]
    win_adv_end   = [k * sigma_adv_steps + Tw_adv_steps for k in range(K)]

    # Correction tensors at full advection resolution (CPU, then moved to device)
    corr_U = torch.zeros(nAdvTotal + 1, Nlat[0], Nlon[0], dtype=dtype)
    corr_V = torch.zeros(nAdvTotal + 1, Nlat[0], Nlon[0], dtype=dtype)

    for t in range(nAdvTotal + 1):
        # ---- All windows whose time interval contains adv step t ----------
        valid_ks = [
            k for k in range(K)
            if win_adv_start[k] <= t <= win_adv_end[k]
        ]
        if not valid_ks:
            # t is not covered by any window → correction remains zero
            continue

        # ---- k* = window whose centre is closest to t  (§3.3) -----------
        k_star = min(
            valid_ks,
            key=lambda k: abs(0.5 * (win_adv_start[k] + win_adv_end[k]) - t),
        )

        # ---- Normalised weights w_k = 1/(|k-k*|+1) (§3.3) --------------
        raw_w = {k: 1.0 / (abs(k - k_star) + 1.0) for k in valid_ks}
        w_sum = sum(raw_w.values())

        cu = torch.zeros(Nlat[0], Nlon[0], dtype=dtype)
        cv = torch.zeros(Nlat[0], Nlon[0], dtype=dtype)
        for k in valid_ks:
            w_k = raw_w[k] / w_sum
            dU_k, dV_k = delta_u_list[k]
            cu += w_k * dU_k.squeeze()
            cv += w_k * dV_k.squeeze()

        corr_U[t] = cu
        corr_V[t] = cv

    # Move correction to device and add to background (Eq. 5)
    Ub_full_int = Ub_full_int.reshape(-1, *Ub_full_int.shape[2:])
    Vb_full_int = Vb_full_int.reshape(-1, *Vb_full_int.shape[2:])
    Uc_full_int = Ub_full_int + corr_U.to(device)[:-1, :, :]
    Vc_full_int = Vb_full_int + corr_V.to(device)[:-1, :, :]

    # ================================================================== #
    # RMS errors: background vs corrected, relative to truth
    # ================================================================== #
    Ut_full_int = Ut_full_int.reshape(-1, *Ut_full_int.shape[2:])
    Vt_full_int = Vt_full_int.reshape(-1, *Vt_full_int.shape[2:])
    RMS_back_list = RMS_per_window_new(
        Ut_full_int, Vt_full_int, Ub_full_int, Vb_full_int
    )
    RMS_corr_list = RMS_per_window_new(
        Ut_full_int, Vt_full_int, Uc_full_int, Vc_full_int
    )

    RMS_back_arr = np.array(RMS_back_list)
    RMS_corr_arr = np.array(RMS_corr_list)

    rms_back_ds = f"RMS_back{suffix}"
    rms_corr_ds = f"RMS_corr{suffix}"
    check_if_exists(rms_back_ds, err_grp)
    check_if_exists(rms_corr_ds, err_grp)
    err_grp.create_dataset(rms_back_ds, data=RMS_back_arr, compression="gzip")
    err_grp.create_dataset(rms_corr_ds, data=RMS_corr_arr, compression="gzip")

    drms          = (RMS_back_arr - RMS_corr_arr) / (RMS_back_arr + 1e-12)
    relative_dRMS = float(np.mean(drms) * 100.0)

    rdelta_ds = f"Relative_Delta_RMS{suffix}"
    check_if_exists(rdelta_ds, err_grp)
    err_grp.create_dataset(rdelta_ds, data=relative_dRMS)

    print(f"\n[sliding_window] Done.  Relative ΔRMS = {relative_dRMS:.4f} %")
    return relative_dRMS, Uc_full_int, Vc_full_int


# ======================================================================== #
# Optional: reconstruct corrected field from already-saved HDF5 corrections
# ======================================================================== #

def reconstruct_from_saved(
    Ub, Vb, assim_grp,
    nDaysSim, lfil, dtAdvection, dtObservation,
    mylr, alpha, beta,
    Tw_days, sigma_days, toll,
    K, device=None,
):
    """
    Phase 2 only — rebuild Eq. 5 from δu_k datasets already stored in HDF5.

    Useful to re-run Phase 2 with different Tw/sigma without re-optimising.

    Parameters
    ----------
    K : int
        Number of windows (must match what was used in Phase 1).

    All other parameters : same as multi_analysis_sliding_window.

    Returns
    -------
    Uc_full_int, Vc_full_int : torch.Tensor
        Corrected velocity field at advection resolution.
    """
    dtype = torch.float32

    nObsStepPerDay  = int(round(24.0 * 3600.0 / dtObservation))
    nAdvStepPerDay  = int(round(24.0 * 3600.0 / dtAdvection))
    sigma_obs_steps = int(round(sigma_days * nObsStepPerDay))
    sigma_adv_steps = int(round(sigma_days * nAdvStepPerDay))
    Tw_adv_steps    = Tw_days * nAdvStepPerDay
    nAdvTotal       = nDaysSim * nAdvStepPerDay

    suffix = _make_suffix(nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days)

    dU_grp = assim_grp["dU"]

    # Load all δu_k from HDF5
    delta_u_list = []
    for k in range(K):
        du_ds = f"dU_win{k + 1:03d}of{K:03d}{suffix}"
        dv_ds = f"dV_win{k + 1:03d}of{K:03d}{suffix}"
        dU_k = torch.from_numpy(dU_grp[du_ds][:]).float()
        dV_k = torch.from_numpy(dU_grp[dv_ds][:]).float()
        delta_u_list.append((dU_k, dV_k))

    # Reconstruct window intervals
    win_adv_start = [k * sigma_adv_steps              for k in range(K)]
    win_adv_end   = [k * sigma_adv_steps + Tw_adv_steps for k in range(K)]

    Nlat_n = Ub.shape[1]
    Nlon_n = Ub.shape[2]

    corr_U = torch.zeros(nAdvTotal + 1, Nlat_n, Nlon_n, dtype=dtype)
    corr_V = torch.zeros(nAdvTotal + 1, Nlat_n, Nlon_n, dtype=dtype)

    for t in range(nAdvTotal + 1):
        valid_ks = [
            k for k in range(K)
            if win_adv_start[k] <= t <= win_adv_end[k]
        ]
        if not valid_ks:
            continue

        k_star = min(
            valid_ks,
            key=lambda k: abs(0.5 * (win_adv_start[k] + win_adv_end[k]) - t),
        )
        raw_w = {k: 1.0 / (abs(k - k_star) + 1.0) for k in valid_ks}
        w_sum = sum(raw_w.values())

        cu = torch.zeros(Nlat_n, Nlon_n, dtype=dtype)
        cv = torch.zeros(Nlat_n, Nlon_n, dtype=dtype)
        for k in valid_ks:
            w_k = raw_w[k] / w_sum
            dU_k, dV_k = delta_u_list[k]
            cu += w_k * dU_k.squeeze()
            cv += w_k * dV_k.squeeze()

        corr_U[t] = cu
        corr_V[t] = cv

    Ub_full_int, Vb_full_int = field_int_new(
        Ub.detach(), Vb.detach(), dtAdvection
    )
    Uc_full_int = Ub_full_int + corr_U.to(device)
    Vc_full_int = Vb_full_int + corr_V.to(device)

    return Uc_full_int, Vc_full_int
