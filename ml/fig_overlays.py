"""True and predicted gain classes for selected test samples (Fig. 5).

Left: true classes; right: predicted classes, where the shade of the
positive cells gives how many of the 12 members predict a gain above the
threshold. Both panels use one threshold, the median of the true and
predicted maps together (clipped at zero), and show the first-day truth
(black) and background (magenta) velocities and the unstable manifold.

Inputs: <output>/evaluation_region_A/sample_XXXX_*/ensemble_prediction.npz
        <output>/deployment_region_A/XXXX_*/manifold/unstable_complete_manifold/
Writes: <output>/figures/overlays/overlay_local_test_XXXX_global_YYYY.pdf,
        classification_colorbar.pdf, velocity_manifold_legend.pdf

Usage:  python ml/fig_overlays.py                  # samples 0, 25 and 82
        python ml/fig_overlays.py --samples 10 11
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

import config as C
from data import test_sample
from ensemble import categorize, classification_metrics
from plotting import CLASS_COLORS, NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR, on_white

ALPHA = 0.6
# Shades of the positive class for <= 6/12, 7/12-9/12 and 10/12-12/12 agreement;
# the middle one is the positive colour as it looks with ALPHA on white.
AGREEMENT_COLORS = ["#C6DBEF", on_white(POSITIVE_COLOR, ALPHA), "#1F5A99"]
AGREEMENT_LEVELS = np.array([0.9, 6.9, 9.9, 12.9]) / 12
QUIVER_SCALE = 5
QUIVER_WIDTH = 0.003
DPI = 250


def class_codes(values, threshold):
    codes = np.ones(values.shape, dtype=np.int8)
    codes[values < 0.0] = 0
    codes[values >= threshold] = 2
    return codes


def plot_sample(index, out_dir):
    name, rec = test_sample(C.REGION_FILES["A"], index)
    pred = np.load(next(C.evaluation_dir("A").glob(f"sample_{index:04d}_*/ensemble_prediction.npz")))
    manifold = np.load(C.deployment_dir("A") / f"{index:04d}_{name}" / "manifold"
                       / "unstable_complete_manifold" / "complete_unstable_manifold.npz")

    y_true, y_pred, members = pred["y_true"], pred["y_pred_mean"], pred["member_predictions"]
    lon, lat = rec["lon"], rec["lat"]
    lon_c, lat_c = lon[C.CROP:-C.CROP], lat[C.CROP:-C.CROP]
    lon_m = manifold["lon_manifold"].reshape(-1)
    lat_m = manifold["lat_manifold"].reshape(-1)
    lon_m, lat_m = lon_m[lon_m != 0], lat_m[lat_m != 0]

    threshold = max(0.0, float(np.quantile(np.concatenate([y_true.ravel(), y_pred.ravel()]),
                                           C.POSITIVE_QUANTILE)))
    agreement = np.mean(members >= threshold, axis=0)
    pred_codes, true_codes = class_codes(y_pred, threshold), class_codes(y_true, threshold)
    accuracy = classification_metrics(categorize(y_true, C.POSITIVE_QUANTILE).positive,
                                      categorize(y_pred, C.POSITIVE_QUANTILE).positive).accuracy

    cmap = ListedColormap(CLASS_COLORS)
    levels = [-0.5, 0.5, 1.5, 2.5]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].contourf(lon_c, lat_c, true_codes, levels=levels, cmap=cmap, alpha=ALPHA, zorder=1)
    axes[1].contourf(lon_c, lat_c, pred_codes, levels=levels, cmap=cmap, alpha=ALPHA, zorder=1)
    axes[1].contourf(lon_c, lat_c, np.ma.masked_where(pred_codes != 2, agreement),
                     levels=AGREEMENT_LEVELS, cmap=ListedColormap(AGREEMENT_COLORS),
                     alpha=1.0, zorder=2)

    for ax in axes:
        ax.plot(lon_m, lat_m, "-k", linewidth=0.5)
        ax.plot(lon_m[0], lat_m[0], "xr", markersize=1)
        ax.quiver(lon, lat, rec["Ut"][0].T, rec["Vt"][0].T, color="k", scale=QUIVER_SCALE,
                  width=QUIVER_WIDTH, alpha=0.5, zorder=5)
        ax.quiver(lon, lat, rec["Ub"][0].T, rec["Vb"][0].T, color="m", scale=QUIVER_SCALE,
                  width=QUIVER_WIDTH, alpha=0.5, zorder=4)
        ax.set_xlabel("Degrees East")
        ax.set_ylabel("Degrees North")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_title(f"Truth\nMaximum improvement: {np.max(y_true):.2f}%")
    axes[1].set_title(f"Predicted\nMaximum improvement: {np.max(y_pred):.2f}%")
    fig.suptitle(f"time = {name[4:14]} to {name[18:]}\nAccuracy: {accuracy * 100}%")

    global_index = C.TEST_START_YEAR * C.SAMPLES_PER_YEAR + index
    out = out_dir / f"overlay_local_test_{index:04d}_global_{global_index:04d}.pdf"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


def save_colorbar_and_legend(out_dir):
    colors = [on_white(NEGATIVE_COLOR, ALPHA), on_white(NEUTRAL_COLOR, ALPHA), *AGREEMENT_COLORS]
    fig, ax = plt.subplots(figsize=(10, 0.2))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(6), 5),
                                              cmap=ListedColormap(colors)),
                        cax=ax, orientation="horizontal", ticks=np.arange(5) + 0.5)
    cbar.ax.set_xticklabels(["Negative", "Neutral", r"$\leq 6/12$", "$7/12$–$9/12$",
                             "$10/12$–$12/12$"])
    cbar.set_label("Improvement classification and predicted ensemble agreement")
    fig.savefig(out_dir / "classification_colorbar.pdf", bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 0.5))
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="magenta", marker=r"$\rightarrow$", linestyle="None",
               markersize=18, label="Background velocity"),
        Line2D([0], [0], color="black", marker=r"$\rightarrow$", linestyle="None",
               markersize=18, label="Truth velocity"),
        Line2D([0], [0], color="black", linewidth=1.5, label="True unstable manifold"),
    ]
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=11,
              handlelength=2.2, columnspacing=2.5)
    fig.savefig(out_dir / "velocity_manifold_legend.pdf", bbox_inches="tight", pad_inches=0.02,
                dpi=DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 25, 82])
    args = parser.parse_args()

    out_dir = C.FIGURES_DIR / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in args.samples:
        plot_sample(index, out_dir)
    save_colorbar_and_legend(out_dir)


if __name__ == "__main__":
    main()
