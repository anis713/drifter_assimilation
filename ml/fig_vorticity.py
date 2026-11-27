"""Mean true gain versus RMS vorticity of the reanalysis (Fig. 11).

For every Region A test sample (the last one excluded, as in the paper),
the relative vorticity is computed on each of its 7 daily truth fields, its
area-weighted (cos latitude) RMS is averaged over the 6 assimilation days,
and compared with the mean of the true gain map.

Writes: <output>/figures/true_vorticity_vs_rdrmse.pdf and prints the
Pearson correlation.

Usage:  python ml/fig_vorticity.py
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import config as C

R_EARTH = 6371000.0
DAYS_PER_SAMPLE = 7
ASSIMILATION_DAYS = 6


def relative_vorticity(U, V, lon, lat):
    """dv/dx - du/dy for fields of shape (time, n, n); axis -2 is
    differentiated along latitude and axis -1 along longitude."""
    lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
    dudy = np.gradient(U, lat_rad, axis=-2) / R_EARTH
    dvdx = np.gradient(V, lon_rad, axis=-1) / (R_EARTH * np.cos(lat_rad)[None, :, None])
    return dvdx - dudy


def rms_vorticity(zeta, lat):
    """Area-weighted RMS over the domain, one value per day."""
    weights = np.cos(np.deg2rad(lat))[None, :, None]
    valid = np.isfinite(zeta)
    num = np.sum(weights * np.where(valid, zeta ** 2, 0.0), axis=(1, 2))
    den = np.sum(weights * valid, axis=(1, 2))
    return np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))


def main():
    with h5py.File(C.REGION_FILES["A"], "r") as f:
        names = sorted(n for n in f.keys() if n.startswith("map_"))
        names = names[C.TEST_START_YEAR * C.SAMPLES_PER_YEAR:-1]
        lon = f[names[0]]["io_map"]["lon"][()]
        lat = f[names[0]]["io_map"]["lat"][()]
        Ut = np.concatenate([f[n]["io_map"]["Ut"][()] for n in names])
        Vt = np.concatenate([f[n]["io_map"]["Vt"][()] for n in names])
        mean_gain = np.array([np.nanmean(f[n]["io_map"]["rdrmse"][()].T) for n in names])

    zeta_rms = rms_vorticity(relative_vorticity(Ut, Vt, lon, lat), lat)
    n_windows = len(zeta_rms) // DAYS_PER_SAMPLE
    zeta_window = 1e5 * np.array([
        np.nanmean(zeta_rms[k * DAYS_PER_SAMPLE:k * DAYS_PER_SAMPLE + ASSIMILATION_DAYS])
        for k in range(n_windows)])

    valid = np.isfinite(zeta_window) & np.isfinite(mean_gain)
    x, y = zeta_window[valid], mean_gain[valid]
    r, p = pearsonr(x, y)
    print(f"{valid.sum()} test samples: Pearson r = {r:.4f} (p = {p:.4e})")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=35, alpha=0.75)
    ax.text(0.05, 0.95, rf"$\rho={r:.3f}$", transform=ax.transAxes, va="top", fontsize=11)
    ax.set_xlabel(r"Reanalysis RMS vorticity ($10^{-5}\ \mathrm{s}^{-1}$)")
    ax.set_ylabel(r"Mean true $\Delta$RMSE (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / "true_vorticity_vs_rdrmse.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {C.FIGURES_DIR / 'true_vorticity_vs_rdrmse.pdf'}")


if __name__ == "__main__":
    main()
