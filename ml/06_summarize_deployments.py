"""Summarise the drifter-release experiments (Figs. 6-9).

The relative improvement of a release at time t is
    100 * (RMSE_background(t) - RMSE_corrected(t)) / RMSE_background(t),
and its mean over the 6-day window is the improvement of that release.

random:         per test sample, the mean and standard deviation over the 50
                releases; mu_t = mean over samples of
                (mean improvement / reference true gain of the sample) * 100,
                with the reference set per region by config.RANDOM_REFERENCE
deterministic:  one release per strategy; mu_t uses the maximum true gain
                of the sample instead of its mean

Writes to <output>/figures/region_<R>_<release>/:
    relative_improvement_per_test_sample.{png,pdf}     (Fig. 6; Fig. 9 for Region B)
    relative_improvement_time_evolution_IQR.{png,pdf}  (Fig. 7)
    predicted_vs_actual_improvement.{png,pdf}          (Fig. 8)

Usage:  python ml/06_summarize_deployments.py --region A --release random
        python ml/06_summarize_deployments.py --region A --release deterministic
        python ml/06_summarize_deployments.py --region B --release random
"""
import argparse
import re
from datetime import datetime

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

import config as C


def relative_improvement(npz):
    """(n_releases, n_time) improvement in %, from one result file."""
    corr = np.atleast_2d(np.squeeze(npz["corr_rmse_t"]))
    back = np.squeeze(npz["rms_back"])
    return 100.0 * (back[None, :] - corr) / back[None, :]


def sample_folders(region):
    folders = [p for p in C.deployment_dir(region).iterdir()
               if p.is_dir() and re.match(r"\d+_map_", p.name)]
    return sorted(folders, key=lambda p: int(p.name.split("_")[0]))


def start_date(folder):
    return datetime.strptime(re.search(r"_map_(\d{4}-\d{2}-\d{2})_to_", folder.name).group(1),
                             "%Y-%m-%d")


def load(region, release):
    """Per-sample quantities of one experiment."""
    model_file, manifold_file = f"cnn_{release}.npz", f"manifold_{release}.npz"
    rows = []
    with h5py.File(C.TRUTH_FILES[region], "r") as truth:
        for folder in sample_folders(region):
            if not ((folder / model_file).exists() and (folder / manifold_file).exists()):
                print(f"Skipping {folder.name}: missing results")
                continue
            model = np.load(folder / model_file)
            rel_r = relative_improvement(model)
            rel_m = relative_improvement(np.load(folder / manifold_file))
            if C.RANDOM_REFERENCE[region] == "positive_region" and release == "random":
                reference = float(model["mean_pos_imp_truth"])
            else:
                group = folder.name.split("_", 1)[1]
                reference = (np.nanmean(truth[group]["io_map"]["rdrmse"][:])
                             if group in truth and "io_map" in truth[group] else np.nan)
            rows.append({
                "date": start_date(folder),
                "model_curve": np.nanmean(rel_r, axis=0),
                "manifold_curve": np.nanmean(rel_m, axis=0),
                # mean over time for each release, then over releases
                "model_mean": np.nanmean(np.nanmean(rel_r, axis=1)),
                "model_std": np.nanstd(np.nanmean(rel_r, axis=1), ddof=1),
                "manifold_mean": np.nanmean(np.nanmean(rel_m, axis=1)),
                "manifold_std": np.nanstd(np.nanmean(rel_m, axis=1), ddof=1),
                "predicted": float(model["mean_pos_imp_pred"] if release == "random"
                                   else model["max_pred"]),
                "true_mean": reference,     # used by the random experiment only
                "true_max": float(model["max_true"]),
            })
    print(f"{len(rows)} test samples")
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def mu_box(ax, mu_manifold, mu_model, loc, anchor, align):
    texts = [TextArea(rf"$\mu_{{t}}(\mathrm{{{name}}}) = {mu:.2f}\%$",
                      textprops=dict(color=color, fontsize=14))
             for name, mu, color in (("manifold", mu_manifold, "tab:blue"),
                                     ("model", mu_model, "tab:orange"))]
    box = AnchoredOffsetbox(loc=loc, child=VPacker(children=texts, align=align, pad=0, sep=6),
                            pad=0.5, borderpad=0.7, frameon=True, bbox_to_anchor=anchor,
                            bbox_transform=ax.transAxes)
    box.patch.set(facecolor="white", edgecolor="white", alpha=1.0, linewidth=1)
    ax.add_artist(box)


def plot_time_series_random(d, out_dir):
    valid = (np.isfinite(d["manifold_mean"]) & np.isfinite(d["model_mean"])
             & np.isfinite(d["true_mean"]) & np.isfinite(d["manifold_std"])
             & np.isfinite(d["model_std"]))
    ratio_ok = valid & (np.abs(d["true_mean"]) > 1e-10)
    mu_manifold = 100 * np.mean(d["manifold_mean"][ratio_ok] / d["true_mean"][ratio_ok])
    mu_model = 100 * np.mean(d["model_mean"][ratio_ok] / d["true_mean"][ratio_ok])

    x = np.arange(valid.sum())
    fig, ax = plt.subplots(figsize=(14, 5))
    for key, color, label in (("manifold", "tab:blue", "Unstable manifold neighborhood"),
                              ("model", "tab:orange", "Model selection")):
        mean, std = d[f"{key}_mean"][valid], d[f"{key}_std"][valid]
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(x, mean, "-o", color=color, linewidth=0.5, markersize=2.8, label=label, zorder=3)
    mu_box(ax, mu_manifold, mu_model, "upper left", (0.02, 1.02), "left")
    ax.plot(x, d["true_mean"][valid], "--", color="tab:green", linewidth=1.35, alpha=0.9,
            label="Mean true positive improvement", zorder=2)
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)

    # About one tick every three months
    ticks, labels, last = [], [], None
    for i, date in enumerate(d["date"][valid]):
        if last is None or (date - last).days >= 90:
            ticks.append(i)
            labels.append(date.strftime("%b %Y"))
            last = date
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Assimilation-window start date")
    ax.set_ylabel("Mean relative improvement over background (%)")
    ax.grid(axis="y", linestyle="-", alpha=0.30)
    ax.legend(loc="upper right", frameon=True, framealpha=1, fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, out_dir, "relative_improvement_per_test_sample", bbox_inches="tight")
    return mu_manifold, mu_model


def plot_time_series_deterministic(d, out_dir):
    mu_manifold = np.mean(d["manifold_mean"] / d["true_max"]) * 100
    mu_model = np.mean(d["model_mean"] / d["true_max"]) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(d["date"], d["manifold_mean"], "o-", markersize=2.8, linewidth=0.5,
            label="Unstable manifold")
    ax.plot(d["date"], d["model_mean"], "s-", markersize=2.8, linewidth=0.5,
            label="Model-selected location")
    ax.plot(d["date"], d["true_max"], "--", markersize=2.8, linewidth=0.8, alpha=0.9, zorder=2,
            label="Maximum truth")
    mu_box(ax, mu_manifold, mu_model, "upper right", (0.9, 1.02), "right")
    ax.axhline(0, color="k", linewidth=1, alpha=0.6)
    ax.set_xlabel("Assimilation-window start date")
    ax.set_ylabel("Relative improvement over background (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", linestyle="-", alpha=0.30)
    ax.legend(framealpha=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, out_dir, "relative_improvement_per_test_sample")
    return mu_manifold, mu_model


def plot_median_evolution(d, release, out_dir):
    """Median over test samples with the 40 %-60 % band."""
    n_time = d["model_curve"].shape[1]
    t = np.arange(n_time) / C.STEPS_PER_DAY
    labels = ({"manifold": "Unstable manifold neighbourhood", "model": "Model selection"}
              if release == "random" else
              {"manifold": "Unstable manifold", "model": "Model-selected location"})

    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("manifold", "model"):
        curves = d[f"{key}_curve"]
        ax.plot(t, np.nanmedian(curves, axis=0), linewidth=2, label=labels[key])
        ax.fill_between(t, np.nanpercentile(curves, 40, axis=0),
                        np.nanpercentile(curves, 60, axis=0), alpha=0.25)
    ax.axhline(0, color="k", linewidth=1, alpha=0.6)
    ax.set_xlabel("Assimilation time (days)")
    ax.set_ylabel("Relative improvement over background (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=1)
    fig.tight_layout()
    save(fig, out_dir, "relative_improvement_time_evolution_IQR")


def plot_predicted_vs_actual(d, release, out_dir):
    ok = np.isfinite(d["predicted"]) & np.isfinite(d["model_mean"])
    predicted, actual = d["predicted"][ok], d["model_mean"][ok]

    lo, hi = np.min(np.concatenate([predicted, actual])), np.max(np.concatenate([predicted, actual]))
    pad = 0.05 * (hi - lo) or 1.0
    lo, hi = lo - pad, hi + pad
    slope, intercept = np.polyfit(predicted, actual, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predicted, actual, s=40, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect prediction")
    ax.plot([lo, hi], slope * np.array([lo, hi]) + intercept, "r-", linewidth=1.5,
            label=f"Linear fit ($y={slope:.2f}x{intercept:+.2f}$)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    if release == "random":
        ax.set_xlabel("Predicted mean improvement (%)")
        ax.set_ylabel("Actual mean improvement (%)")
    else:
        ax.set_xlabel("Model-predicted maximum improvement (%)")
        ax.set_ylabel("Actual improvement at model-selected location (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=1)
    fig.tight_layout()
    save(fig, out_dir, "predicted_vs_actual_improvement")

    error = actual - predicted
    r = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else np.nan
    print(f"Predicted vs actual: MAE {np.mean(np.abs(error)):.3f}, "
          f"RMSE {np.sqrt(np.mean(error ** 2)):.3f}, bias {np.mean(error):.3f} "
          f"percentage points, Pearson r {r:.3f}, fit y = {slope:.2f}x {intercept:+.2f}")


def save(fig, out_dir, name, **kwargs):
    fig.savefig(out_dir / f"{name}.png", dpi=300, **kwargs)
    fig.savefig(out_dir / f"{name}.pdf", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", choices=C.REGION_FILES, default="A")
    parser.add_argument("--release", choices=["random", "deterministic"], required=True)
    args = parser.parse_args()

    out_dir = C.FIGURES_DIR / f"region_{args.region}_{args.release}"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load(args.region, args.release)

    if args.release == "random":
        mu_manifold, mu_model = plot_time_series_random(d, out_dir)
    else:
        mu_manifold, mu_model = plot_time_series_deterministic(d, out_dir)
    plot_median_evolution(d, args.release, out_dir)
    plot_predicted_vs_actual(d, args.release, out_dir)

    print(f"Mean improvement over time: model {np.mean(d['model_mean']):.3f} %, "
          f"manifold {np.mean(d['manifold_mean']):.3f} %")
    print(f"mu_t(model) = {mu_model:.2f} %, mu_t(manifold) = {mu_manifold:.2f} %")
    print(f"Figures in {out_dir}")


if __name__ == "__main__":
    main()
