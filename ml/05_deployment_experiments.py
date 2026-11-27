"""Drifter-release experiments on the test samples.

For one test sample, a synthetic drifter is released, its trajectory is
simulated on the reanalysis (truth) and assimilated into the altimetry
(background) with the sliding-window method of src/. The RMSE of the
corrected field is compared with that of the background over the 6-day
window. Four release strategies are compared:

  random experiment (50 releases each; each strategy re-seeded with RANDOM_SEED):
    cnn_random              uniformly at random in the region the ensemble
                            predicts as positive
    manifold_random         uniformly at random within MANIFOLD_OFFSET deg of
                            the unstable manifold
  deterministic experiment (one release each):
    cnn_deterministic       at the maximum of the predicted gain map
    manifold_deterministic  at the most hyperbolic point of the unstable
                            manifold (smallest det J)

The unstable manifold is computed on the first-day truth field held fixed
in time (src/manifolds.py).

Inputs: <output>/evaluation_region_<R>/sample_XXXX_*/ensemble_prediction.npz
        (from 04_evaluate_ensemble.py)
Writes to <output>/deployment_region_<R>/XXXX_<group>/:
    manifold/               candidate manifolds, Best/my_unstable.npz and
                            unstable_complete_manifold/complete_unstable_manifold.npz
    cnn_random.npz, manifold_random.npz, cnn_deterministic.npz,
    manifold_deterministic.npz
        all_rmse     relative RMSE reduction of each release (%)
        corr_rmse_t  RMSE time series of the corrected field, (n_releases, n_time)
        rms_back     RMSE time series of the background, (n_time,)
        lon0, lat0   release positions
        plus the predicted / true gains used by 06_summarize_deployments.py

Each test sample is independent, so on a cluster run one sample per job:
    python ml/05_deployment_experiments.py --region A --sample $SLURM_ARRAY_TASK_ID
Or all samples in sequence:
    python ml/05_deployment_experiments.py --region A --all
"""
import argparse
import os
import tempfile

import h5py
import numpy as np
import torch

import config as C
from data import test_sample
from analysis_nDrif import run_onTruth_new
from adv_torch import simul_drifter
from manifolds import Jacob_eval, all_initial, best_manifold
from sliding_window_assim import _make_suffix, multi_analysis_sliding_window

A = C.ASSIMILATION
CROP = C.CROP


# ---------------------------------------------------------------------------
# One assimilation experiment
# ---------------------------------------------------------------------------

def assimilate(rec, lon0, lat0, work_dir):
    """Release drifters at (lon0, lat0) and assimilate their trajectories.

    Returns (relative RMSE reduction in %, background RMSE series,
    corrected RMSE series).
    """
    maskuu, maskvv = np.isnan(rec["Ut"][0]), np.isnan(rec["Vt"][0])
    Ut, Vt = torch.tensor(rec["Ut"]), torch.tensor(rec["Vt"])
    Ub, Vb = torch.tensor(rec["Ub"]), torch.tensor(rec["Vb"])
    lon, lat = torch.tensor(rec["lon"]), torch.tensor(rec["lat"])

    fd, path = tempfile.mkstemp(suffix=".h5", dir=work_dir)
    os.close(fd)
    os.remove(path)
    try:
        with h5py.File(path, "a") as h5:
            obs_grp = h5.require_group("metadata")
            run_onTruth_new(Ut, Vt, lon, lat, A["nDaysSim"], obs_grp, A["dtAdvection"],
                            lon0, lat0, A["n_drif"])
            rel_drms, _, _ = multi_analysis_sliding_window(
                Ut, Vt, Ub, Vb, lon, lat, lon0, lat0, maskuu, maskvv, h5, obs_grp,
                nDaysSim=A["nDaysSim"], lfil=A["lfil"], dtAdvection=A["dtAdvection"],
                dtObservation=A["dtObservation"], mylr=A["mylr"], alpha=A["alpha"],
                beta=A["beta"], n_drif=A["n_drif"], Tw_days=A["Tw_days"],
                sigma_days=A["sigma_days"], toll=A["toll"])
            suffix = _make_suffix(A["nDaysSim"], A["lfil"], A["alpha"], A["beta"], A["toll"],
                                  A["mylr"], A["Tw_days"], A["sigma_days"])
            rms_back = h5[f"errors/RMS_back{suffix}"][:]
            rms_corr = h5[f"errors/RMS_corr{suffix}"][:]
    finally:
        if os.path.exists(path):
            os.remove(path)
    return rel_drms, rms_back, rms_corr


def run_releases(rec, lon0_all, lat0_all, work_dir):
    all_rmse, corr = [], []
    for k, (lon0, lat0) in enumerate(zip(lon0_all, lat0_all), start=1):
        rel, rms_back, rms_corr = assimilate(rec, [lon0], [lat0], work_dir)
        all_rmse.append(rel)
        corr.append(rms_corr)
        print(f"  release {k}/{len(lon0_all)}: {rel:.3f} %")
    return dict(all_rmse=np.array(all_rmse), corr_rmse_t=np.array(corr),
                rms_back=np.array(rms_back), lon0=np.array(lon0_all), lat0=np.array(lat0_all))


# ---------------------------------------------------------------------------
# Unstable manifold of the first-day truth field
# ---------------------------------------------------------------------------

def unstable_manifold(rec, out_dir):
    """Best unstable manifold and its full trajectory (computed once per sample)."""
    manifold_dir = out_dir / "manifold"
    complete_dir = manifold_dir / "unstable_complete_manifold"
    complete_file = complete_dir / "complete_unstable_manifold.npz"
    best_file = manifold_dir / "Best" / "my_unstable.npz"
    if complete_file.exists() and best_file.exists():
        return best_file, complete_file

    complete_dir.mkdir(parents=True, exist_ok=True)
    lon, lat = rec["lon"], rec["lat"]
    Ut_const = np.repeat(rec["Ut"][0:1], 7, axis=0)
    Vt_const = np.repeat(rec["Vt"][0:1], 7, axis=0)
    c = slice(CROP, -CROP)

    if not list(manifold_dir.glob("*.npz")):
        all_initial([lon[0], lon[-1]], [lat[0], lat[-1]], Ut_const[:, c, c], Vt_const[:, c, c],
                    lon[c], lat[c], A["nDaysSim"], A["dtAdvection"], saving_dir=manifold_dir)
    best_file = best_manifold(manifold_dir, saving_dir=manifold_dir)

    start = np.load(best_file)
    lon_m, lat_m = simul_drifter(torch.tensor([start["this_lon"].reshape(-1)[0]]),
                                 torch.tensor([start["this_lat"].reshape(-1)[0]]),
                                 torch.tensor(Ut_const), torch.tensor(Vt_const),
                                 torch.tensor(lon), torch.tensor(lat), A["dtAdvection"])
    np.savez(complete_file, lon_manifold=lon_m, lat_manifold=lat_m)
    return best_file, complete_file


# ---------------------------------------------------------------------------
# Release strategies
# ---------------------------------------------------------------------------

def cnn_random(rec, pred, out_dir):
    pred_map = pred["y_pred_mean"]
    good = pred_map > pred["predicted_positive_threshold"]
    true_map = rec["true_map"]
    lon, lat = rec["lon"], rec["lat"]

    # Coordinates of the predicted-positive cells, and the acceptance rule of
    # the published runs: a point is kept when each of the four surrounding
    # grid nodes has its longitude among Lon_good and its latitude among Lat_good.
    Lon, Lat = np.meshgrid(lon[CROP:-CROP], lat[CROP:-CROP])
    Lon_good, Lat_good = Lon[good], Lat[good]

    lon0_all, lat0_all = [], []
    while len(lon0_all) < C.N_RANDOM_RELEASES:
        lon_p = np.random.uniform(lon[CROP], lon[-CROP])
        lat_p = np.random.uniform(lat[CROP], lat[-CROP])
        i = np.searchsorted(lon, lon_p) - 1
        j = np.searchsorted(lat, lat_p) - 1
        corners = [(lon[i], lat[j]), (lon[i + 1], lat[j]), (lon[i], lat[j + 1]),
                   (lon[i + 1], lat[j + 1])]
        if all(cl.item() in Lon_good and ct.item() in Lat_good for cl, ct in corners):
            lon0_all.append(lon_p)
            lat0_all.append(lat_p)

    print("CNN-guided random releases")
    result = run_releases(rec, lon0_all, lat0_all, out_dir)
    np.savez(out_dir / "cnn_random.npz", **result,
             mean_pos_imp_pred=pred_map[good].mean(),
             mean_pos_imp_truth=true_map[good].mean(),
             max_true=np.max(true_map), mean_true=np.mean(true_map))


def manifold_random(rec, complete_file, out_dir):
    m = np.load(complete_file)
    lon_m = m["lon_manifold"].reshape(-1)
    lat_m = m["lat_manifold"].reshape(-1)
    lon_m, lat_m = lon_m[lon_m != 0], lat_m[lat_m != 0]

    lon0_all, lat0_all = [], []
    while len(lon0_all) < C.N_RANDOM_RELEASES:
        lon_p = np.random.uniform(lon_m[0], lon_m[-1])
        lat_p = np.random.uniform(lat_m[0], lat_m[-1])
        if any(abs(lon_p - a.item()) < C.MANIFOLD_OFFSET and abs(lat_p - b.item()) < C.MANIFOLD_OFFSET
               for a, b in zip(lon_m, lat_m)):
            lon0_all.append(lon_p)
            lat0_all.append(lat_p)

    print("Manifold-guided random releases")
    np.savez(out_dir / "manifold_random.npz", **run_releases(rec, lon0_all, lat0_all, out_dir))


def cnn_deterministic(rec, pred, out_dir):
    pred_map = pred["y_pred_mean"]
    i, j = np.unravel_index(np.argmax(pred_map), pred_map.shape)
    lon0, lat0 = rec["lon"][i + CROP], rec["lat"][j + CROP]

    print("CNN-guided single release")
    np.savez(out_dir / "cnn_deterministic.npz", **run_releases(rec, [lon0], [lat0], out_dir),
             max_pred=pred_map[i, j], max_true=np.max(rec["true_map"]))


def manifold_deterministic(rec, best_file, jacobian_rec, out_dir):
    """Release at the point of the manifold with the smallest det(J).

    det(J) is evaluated on the first-day truth field of ``jacobian_rec``
    (test sample 0 of the region, as in the published runs).
    """
    m = np.load(best_file)
    lon_m, lat_m = m["this_lon"], m["this_lat"]
    lon = torch.tensor(jacobian_rec["lon"])
    lat = torch.tensor(jacobian_rec["lat"])
    U0 = torch.tensor(jacobian_rec["Ut"])[0]
    V0 = torch.tensor(jacobian_rec["Vt"])[0]

    min_det, min_idx = 1, -1
    for k in range(len(lon_m)):
        det = torch.det(Jacob_eval(lon, lat, U0, V0, torch.tensor([lon_m[k]]),
                                   torch.tensor([lat_m[k]]))).item()
        if det < min_det:
            min_det, min_idx = det, k

    print("Manifold-guided single release")
    np.savez(out_dir / "manifold_deterministic.npz",
             **run_releases(rec, [lon_m[min_idx]], [lat_m[min_idx]], out_dir))


# ---------------------------------------------------------------------------

def run_sample(region, index, experiments):
    h5_path = C.REGION_FILES[region]
    name, rec = test_sample(h5_path, index)
    out_dir = C.deployment_dir(region) / f"{index:04d}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = next(C.evaluation_dir(region).glob(f"sample_{index:04d}_*/ensemble_prediction.npz"))
    pred = np.load(pred_file)
    print(f"=== Region {region}, test sample {index}: {name} -> {out_dir}")

    best_file, complete_file = unstable_manifold(rec, out_dir)

    if "random" in experiments:
        np.random.seed(C.RANDOM_SEED)       # each random experiment starts from the same seed
        cnn_random(rec, pred, out_dir)
        np.random.seed(C.RANDOM_SEED)
        manifold_random(rec, complete_file, out_dir)
    if "deterministic" in experiments:
        cnn_deterministic(rec, pred, out_dir)
        _, jacobian_rec = test_sample(h5_path, 0)
        manifold_deterministic(rec, best_file, jacobian_rec, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", choices=C.REGION_FILES, default="A")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--sample", type=int, help="local index of one test sample")
    which.add_argument("--all", action="store_true",
                       help="every test sample of the paper in turn (DEPLOYMENT_SAMPLES)")
    parser.add_argument("--experiments", nargs="+", choices=["random", "deterministic"],
                        default=["random", "deterministic"])
    args = parser.parse_args()

    indices = range(C.DEPLOYMENT_SAMPLES[args.region]) if args.all else [args.sample]
    for index in indices:
        run_sample(args.region, index, args.experiments)


if __name__ == "__main__":
    main()
