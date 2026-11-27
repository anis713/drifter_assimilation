"""Unstable manifolds of a velocity field (baseline release strategy).

Algorithm (Jacobian criterion, as in Molcard et al. 2006):
1. find quasi-fixed points, where both time-mean velocity components are
   smaller than ``cutoff`` (m/s);
2. around each fixed point, advect every grid point backward in time while
   det(J) < 0 (hyperbolic region) and keep the longest trajectory;
3. the longest of these trajectories approximates the unstable manifold.

All velocity arrays are indexed (time, lon, lat).

Public functions:
    all_initial     candidate unstable manifolds of a field, saved as .npz
    best_manifold   longest candidate, saved as Best/my_<type>.npz
    Jacob_eval      Jacobian of (u, v) at one point
"""
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from multilinear import RegularGridInterpolator

R_earth = 6371229
resolution = 0.125     # grid spacing (deg) used for the finite differences
tol = -1e-12           # det(J) < tol means hyperbolic


# ---------------------------------------------------------------------------
# Geometry, interpolation and advection
# ---------------------------------------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance (m)."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return R_earth * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def myinterpolate(U, lon, lat, mylon, mylat):
    interp = RegularGridInterpolator((lon, lat), U)
    return interp(torch.stack([mylon, mylat]))


def timeinterpolate(U1, U2, V1, V2, t):
    """Linear interpolation between two daily fields, t in seconds after day 1."""
    return U1 + (U2 - U1) * t / (24 * 3600), V1 + (V2 - V1) * t / (24 * 3600)


def fromMetersToDeg(du, dv):
    dun = (180 / math.pi) * (du / (R_earth * math.cos(du / R_earth)))
    dvn = (180 / math.pi) * (dv / R_earth)
    return dun, dvn


def fromDegToMeters(mylon, mylat):
    phi = math.radians(mylat)
    theta = math.radians(mylon)
    return R_earth * theta * math.cos(phi), R_earth * phi


def interpolate2drifter(t, mylon, mylat, U1, U2, V1, V2, lon, lat):
    u1 = myinterpolate(U1, lon, lat, mylon, mylat)
    u2 = myinterpolate(U2, lon, lat, mylon, mylat)
    v1 = myinterpolate(V1, lon, lat, mylon, mylat)
    v2 = myinterpolate(V2, lon, lat, mylon, mylat)
    return timeinterpolate(u1, u2, v1, v2, t)


def RK4(mylonIN, mylatIN, dtAdvection, t0, U11, U22, V11, V22, lon, lat):
    """One fourth-order Runge-Kutta step of length dtAdvection (s)."""
    fields = dict(U1=U11, U2=U22, V1=V11, V2=V22, lon=lon, lat=lat)

    u1, v1 = interpolate2drifter(t0, mylonIN, mylatIN, **fields)
    dun1, dvn1 = fromMetersToDeg(dtAdvection * u1, dtAdvection * v1)

    u2, v2 = interpolate2drifter(t0 + 0.5 * dtAdvection, mylonIN + 0.5 * dun1,
                                 mylatIN + 0.5 * dvn1, **fields)
    dun2, dvn2 = fromMetersToDeg(dtAdvection * u2, dtAdvection * v2)

    u3, v3 = interpolate2drifter(t0 + 0.5 * dtAdvection, mylonIN + 0.5 * dun2,
                                 mylatIN + 0.5 * dvn2, **fields)
    dun3, dvn3 = fromMetersToDeg(dtAdvection * u3, dtAdvection * v3)

    u4, v4 = interpolate2drifter(t0 + dtAdvection, mylonIN + dun3, mylatIN + dvn3, **fields)
    dun4, dvn4 = fromMetersToDeg(dtAdvection * u4, dtAdvection * v4)

    return (mylonIN + (dun1 + 2 * dun2 + 2 * dun3 + dun4) / 6,
            mylatIN + (dvn1 + 2 * dvn2 + 2 * dvn3 + dvn4) / 6)


def avgVfield(Ut, Vt, nDaysSim):
    """Time-mean field, returned indexed (lat, lon)."""
    return (torch.sum(torch.transpose(Ut, 1, 2), dim=0) / nDaysSim,
            torch.sum(torch.transpose(Vt, 1, 2), dim=0) / nDaysSim)


def forward_simul(mylonIN, mylatIN, Ut, Vt, lon, lat, dtAdvection):
    """Forward trajectory over the whole field; returns (days-1, steps) lon/lat."""
    steps_per_day = int(24.0 * 3600.0 / dtAdvection)
    n_days = Ut.shape[0]
    store_lon = torch.empty((n_days - 1, steps_per_day))
    store_lat = torch.empty((n_days - 1, steps_per_day))
    store_lon[0][0] = mylonIN
    store_lat[0][0] = mylatIN

    for day in tqdm(range(n_days - 1)):
        for step in range(steps_per_day):
            mylonIN, mylatIN = RK4(mylonIN, mylatIN, dtAdvection, step * dtAdvection,
                                   U11=Ut[day], U22=Ut[day + 1], V11=Vt[day], V22=Vt[day + 1],
                                   lon=lon, lat=lat)
            store_lon[day][step] = mylonIN[0]
            store_lat[day][step] = mylatIN[0]
    return store_lon, store_lat


# ---------------------------------------------------------------------------
# Jacobian criterion
# ---------------------------------------------------------------------------

def find_fixedpts(lon, lat, U, V, nDaysSim, cutoff):
    """Grid points where both time-mean velocity components are below cutoff."""
    Ut, Vt = avgVfield(U, V, nDaysSim)
    lon_fixed, lat_fixed = [], []
    for ilon in lon.tolist():
        for jlat in lat.tolist():
            u = myinterpolate(Ut.t(), lon, lat, torch.tensor([ilon]), torch.tensor([jlat]))
            v = myinterpolate(Vt.t(), lon, lat, torch.tensor([ilon]), torch.tensor([jlat]))
            if abs(u) < cutoff and u != 0 and abs(v) < cutoff and v != 0:
                lon_fixed.append(ilon)
                lat_fixed.append(jlat)
    return lon_fixed, lat_fixed


def Jacob_eval(lon, lat, U, V, mylon, mylat):
    """2x2 Jacobian [[du/dx, du/dy], [dv/dx, dv/dy]] at (mylon, mylat), by
    centred differences of +/- ``resolution`` degrees. U, V: (lon, lat)."""
    un_lon = myinterpolate(U, lon, lat, mylon + resolution, mylat)
    up_lon = myinterpolate(U, lon, lat, mylon - resolution, mylat)
    un_lat = myinterpolate(U, lon, lat, mylon, mylat + resolution)
    up_lat = myinterpolate(U, lon, lat, mylon, mylat - resolution)
    vn_lon = myinterpolate(V, lon, lat, mylon + resolution, mylat)
    vp_lon = myinterpolate(V, lon, lat, mylon - resolution, mylat)
    vn_lat = myinterpolate(V, lon, lat, mylon, mylat + resolution)
    vp_lat = myinterpolate(V, lon, lat, mylon, mylat - resolution)

    x, y = fromDegToMeters(mylon, mylat)
    xn, yn = fromDegToMeters(mylon + resolution, mylat + resolution)
    xp, yp = fromDegToMeters(mylon - resolution, mylat - resolution)
    inv_dx = 1 / ((xn - x) + (x - xp))
    inv_dy = 1 / ((yn - y) + (y - yp))

    return torch.tensor([[(un_lon - up_lon) * inv_dx, (un_lat - up_lat) * inv_dy],
                         [(vn_lon - vp_lon) * inv_dx, (vn_lat - vp_lat) * inv_dy]])


def conditioned_backward_simul(lon, lat, Ut, Vt, mylonIN, mylatIN, dtAdvection, nDaysSim):
    """Advect backward in time as long as det(J) < 0 along the trajectory.

    Returns (time spent in the hyperbolic region in s, lon, lat) with the
    trajectory stored as (nDaysSim, steps) arrays, zero after it stops.
    """
    steps_per_day = int(24.0 * 3600.0 / dtAdvection)
    store_lon = torch.zeros((nDaysSim, steps_per_day))
    store_lat = torch.zeros((nDaysSim, steps_per_day))
    store_lon[0][0] = mylonIN
    store_lat[0][0] = mylatIN
    last = 0

    for day in tqdm(range(nDaysSim - 1, 0, -1)):
        U1, V1 = Ut[day], Vt[day]
        U2, V2 = Ut[day - 1], Vt[day - 1]
        if not torch.det(Jacob_eval(lon, lat, U1, V1, mylonIN, mylatIN)) < tol:
            break
        for step in range(steps_per_day):
            t0 = step * dtAdvection
            mylonIN, mylatIN = RK4(mylonIN, mylatIN, dtAdvection, t0,
                                   U11=-U1, U22=-U2, V11=-V1, V22=-V2, lon=lon, lat=lat)
            U_int, V_int = timeinterpolate(-U1, -U2, -V1, -V2, t0)
            if torch.det(Jacob_eval(lon, lat, U_int, V_int, mylonIN, mylatIN)) < tol:
                store_lon[nDaysSim - 1 - day][step] = mylonIN[0]
                store_lat[nDaysSim - 1 - day][step] = mylatIN[0]
                last = (nDaysSim - 1 - day) * 24.0 * 3600.0 + t0
            else:
                break
    return last, store_lon, store_lat


def nbhd_simul(lon, lat, U, V, LON, LAT, mylon, mylat, dtAdvection, nDaysSim, err):
    """Longest backward hyperbolic trajectory started from the grid points in
    the box of half-width ``err`` (deg) around (mylon, mylat), clipped to
    LON x LAT. Returns (time, start lon, start lat, trajectory lon, lat)."""
    lon_limit = [max(mylon - err, LON[0]), min(mylon + err, LON[1])]
    lat_limit = [max(mylat - err, LAT[0]), min(mylat + err, LAT[1])]
    masklon = (lon >= lon_limit[0]) & (lon < lon_limit[1])
    masklat = (lat >= lat_limit[0]) & (lat < lat_limit[1])
    lon, lat = lon[masklon], lat[masklat]

    Ut = torch.zeros((nDaysSim, len(lon), len(lat)))
    Vt = torch.zeros((nDaysSim, len(lon), len(lat)))
    for day in range(nDaysSim):
        Ut[day] = U[day][masklon][:, masklat]
        Vt[day] = V[day][masklon][:, masklat]

    best_time, my_lon, my_lat = 0, torch.empty(0), torch.empty(0)
    lon_trj, lat_trj = torch.empty(0), torch.empty(0)
    for ilon in lon.tolist():
        for jlat in lat.tolist():
            start_lon, start_lat = torch.tensor([ilon]), torch.tensor([jlat])
            time, stored_lon, stored_lat = conditioned_backward_simul(
                lon, lat, Ut, Vt, start_lon, start_lat, dtAdvection, nDaysSim)
            if time > best_time:
                best_time = time
                lon_trj, lat_trj = stored_lon.clone(), stored_lat.clone()
                my_lon, my_lat = start_lon.clone(), start_lat.clone()
    return best_time, my_lon, my_lat, lon_trj, lat_trj


def unstable_manifolds(LON, LAT, lon, lat, U, V, dtAdvection, nDaysSim, cutoff, err,
                       out_dir, plot=False):
    """Save one candidate unstable manifold per (separated) fixed point as
    ``Unst_manifold<k>_long<seconds>.npz`` (keys st_lon, st_lat)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    lon_fixed, lat_fixed = find_fixedpts(lon, lat, U, V, nDaysSim, cutoff)

    if plot:
        avgU, avgV = avgVfield(U, V, nDaysSim)
        Lon, Lat = np.meshgrid(lon, lat)
        plt.figure()
        plt.quiver(Lon, Lat, avgU, avgV, scale=5)

    prev_lon = prev_lat = 0
    for counter, (ifx, jfx) in enumerate(zip(lon_fixed, lat_fixed)):
        if plot:
            plt.plot(ifx, jfx, "x", color="green", markersize=5)
        # Skip a fixed point that is too close to the previous one (distance
        # in metres compared with 2*err, as in the published runs).
        if counter == 0 or haversine(prev_lon, prev_lat, ifx, jfx) > 2 * err:
            print(f"sub-block: {counter + 1}/{len(lon_fixed)}")
            time, start_lon, start_lat, lon_trj, lat_trj = nbhd_simul(
                lon, lat, U, V, LON, LAT, ifx, jfx, dtAdvection, nDaysSim, err)
            if time != 0:
                np.savez(out_dir / f"Unst_manifold{counter + 1}_long{time:.0f}.npz",
                         st_lon=lon_trj.detach().numpy(), st_lat=lat_trj.detach().numpy())
                if plot:
                    trj_lon, trj_lat = forward_simul(start_lon, start_lat, U, V, lon, lat,
                                                     dtAdvection)
                    trj_lon = trj_lon[:nDaysSim - 1].flatten()
                    trj_lat = trj_lat[:nDaysSim - 1].flatten()
                    inside = ((trj_lon > max(ifx - err, LON[0])) & (trj_lon < min(ifx + err, LON[1]))
                              & (trj_lat > max(jfx - err, LAT[0])) & (trj_lat < min(jfx + err, LAT[1])))
                    plt.plot(trj_lon[inside], trj_lat[inside], "o", color="red", markersize=0.25)
        prev_lon, prev_lat = ifx, jfx

    if plot:
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(out_dir / "figure_initial_manifolds.png")
        plt.close()


# ---------------------------------------------------------------------------
# Entry points used by ml/05_deployment_experiments.py
# ---------------------------------------------------------------------------

def all_initial(LON, LAT, U, V, lon, lat, nDaysSim, dtAdvection, saving_dir):
    """Candidate unstable manifolds of the field U, V (numpy, (time, lon, lat))."""
    unstable_manifolds(LON, LAT, lon=torch.tensor(lon), lat=torch.tensor(lat),
                       U=torch.tensor(U, dtype=torch.float32),
                       V=torch.tensor(V, dtype=torch.float32),
                       dtAdvection=dtAdvection, nDaysSim=nDaysSim, cutoff=0.15, err=0.3,
                       out_dir=saving_dir, plot=True)


def _natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def best_manifold(manifolds_path, saving_dir, manifold_type="unstable"):
    """Keep the candidate with the most stored positions (the longest one).

    Saves its non-zero positions to <saving_dir>/Best/my_<type>.npz
    (keys this_lon, this_lat) and returns that path.
    """
    candidates = sorted(Path(manifolds_path).glob("*.npz"), key=_natural_key)
    best, best_len = None, -1
    for path in candidates:
        st_lon = np.load(path)["st_lon"].reshape(-1)
        n = st_lon[st_lon != 0].shape[0]
        if best_len < n:
            best, best_len = path, n

    manifold = np.load(best)
    lon_trj = manifold["st_lon"].reshape(-1)
    lat_trj = manifold["st_lat"].reshape(-1)

    out = Path(saving_dir) / "Best"
    out.mkdir(exist_ok=True)
    file_name = out / f"my_{manifold_type}.npz"
    np.savez(file_name, this_lon=lon_trj[lon_trj != 0], this_lat=lat_trj[lat_trj != 0])
    return file_name
