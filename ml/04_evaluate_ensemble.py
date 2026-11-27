"""Predict the gain map of every test sample with the 12-member ensemble.

For each test sample, the prediction is the mean of the 12 members. The
script computes regression scores and the classification of cells into
negative / neutral / positive gain, and saves the predictions used by the
drifter-release experiments.

Writes to <output>/evaluation_region_<R>/:
    sample_XXXX_<group>/ensemble_prediction.npz   y_true, y_pred_mean, y_pred_std,
                                                  member_predictions and both thresholds
    sample_XXXX_<group>/true_pred_error.png, ensemble_std.png
    per_sample_metrics.csv, per_model_test_losses.csv, summary.{json,csv}

Usage:  python ml/04_evaluate_ensemble.py --region A
        python ml/04_evaluate_ensemble.py --region B
"""
import argparse
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from torch.utils.data import Subset

import config as C
from data import VelocityDataset, load_records, test_indices
from ensemble import (categorize, classification_metrics, ensemble_checkpoints,
                      load_ensemble, metrics_from_counts, predict_ensemble,
                      regression_metrics)
from plotting import CLASS_COLORS

QUIVER_SCALE = 7
QUIVER_WIDTH = 0.003
DPI = 200


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(ax, lon, lat, flow, values, categories, title):
    """Classes of one gain map with the time-mean truth (black) and background
    (magenta) velocities of the sample and the following one."""
    lon_c, lat_c = lon[C.CROP:-C.CROP], lat[C.CROP:-C.CROP]
    contour = ax.contourf(lon_c, lat_c, categories.codes, levels=[-0.5, 0.5, 1.5, 2.5],
                          cmap=ListedColormap(CLASS_COLORS), alpha=0.55, zorder=1)
    for (u, v), color in ((flow["truth"], "k"), (flow["background"], "m")):
        ax.quiver(lon, lat, np.mean(u, axis=0).T, np.mean(v, axis=0).T, color=color,
                  scale=QUIVER_SCALE, width=QUIVER_WIDTH, zorder=5)
    ax.set_title(f"{title}\n($g_{{\\max}}$={np.max(values):.3f}, "
                 f"$g^*$={categories.threshold:.3f})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(contour, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["negative effect", "neutral effect", "positive effect"])


def save_sample_figures(folder, lon, lat, flow, y_true, y_pred, y_std, cat_true, cat_pred,
                        cls, reg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    fig.suptitle(f"{C.N_ENSEMBLE}-model bagged prediction | "
                 f"Accuracy={100 * cls.accuracy:.1f}% | F1={cls.f1:.3f} | "
                 f"SmoothL1={reg['smooth_l1']:.4f}", fontsize=13)
    plot_panel(axes[0], lon, lat, flow, y_true, cat_true, "Truth")
    plot_panel(axes[1], lon, lat, flow, y_pred, cat_pred, "Bagged prediction")
    fig.savefig(folder / "true_pred_error.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    contour = ax.contourf(lon[C.CROP:-C.CROP], lat[C.CROP:-C.CROP], y_std, levels=20)
    plt.colorbar(contour, ax=ax, label=f"Standard deviation across {C.N_ENSEMBLE} models")
    ax.set_title(f"Ensemble disagreement\nmean std={np.mean(y_std):.4f}, "
                 f"max std={np.max(y_std):.4f}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(folder / "ensemble_std.png", dpi=DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", choices=C.REGION_FILES, default="A")
    args = parser.parse_args()

    h5_path = C.REGION_FILES[args.region]
    out_dir = C.evaluation_dir(args.region)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = VelocityDataset(h5_path, crop=C.CROP)
    test_idx = test_indices(len(dataset))
    test_set = Subset(dataset, test_idx.tolist())
    names, records = load_records(h5_path)
    test_names = [names[i] for i in test_idx]
    test_records = [records[i] for i in test_idx]
    print(f"Region {args.region}: {len(test_idx)} test samples "
          f"({test_names[0]} ... {test_names[-1]})")

    checkpoints = ensemble_checkpoints()
    models = load_ensemble(checkpoints)

    rows, member_losses = [], [[] for _ in models]
    totals = np.zeros(4, dtype=int)    # tp, tn, fp, fn

    for i, (x, y) in enumerate(test_set):
        y_true = y.squeeze(0).numpy()
        y_pred, y_std, members = predict_ensemble(models, x)
        reg = regression_metrics(y_true, y_pred)
        cat_true = categorize(y_true, C.POSITIVE_QUANTILE)
        cat_pred = categorize(y_pred, C.POSITIVE_QUANTILE)
        cls = classification_metrics(cat_true.positive, cat_pred.positive)
        totals += (cls.tp, cls.tn, cls.fp, cls.fn)

        losses = [regression_metrics(y_true, m)["smooth_l1"] for m in members]
        for k, loss in enumerate(losses):
            member_losses[k].append(loss)

        name = test_names[i]
        folder = out_dir / f"sample_{i:04d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}"
        folder.mkdir(exist_ok=True)

        # Velocities of this sample and the next one, for the plotted time mean
        window = [test_records[i]] + ([test_records[i + 1]] if i + 1 < len(test_records) else [])
        flow = {
            "truth": (np.concatenate([r["Ut"] for r in window]),
                      np.concatenate([r["Vt"] for r in window])),
            "background": (np.concatenate([r["Ub"] for r in window]),
                           np.concatenate([r["Vb"] for r in window])),
        }
        save_sample_figures(folder, test_records[i]["lon"], test_records[i]["lat"], flow,
                            y_true, y_pred, y_std, cat_true, cat_pred, cls, reg)
        np.savez_compressed(folder / "ensemble_prediction.npz",
                            y_true=y_true, y_pred_mean=y_pred, y_pred_std=y_std,
                            member_predictions=members,
                            true_positive_threshold=cat_true.threshold,
                            predicted_positive_threshold=cat_pred.threshold)

        rows.append({
            "local_test_index": i, "h5_group": name, **reg,
            "accuracy": cls.accuracy, "precision": cls.precision, "recall": cls.recall,
            "f1": cls.f1, "tp": cls.tp, "tn": cls.tn, "fp": cls.fp, "fn": cls.fn,
            "true_threshold": cat_true.threshold, "pred_threshold": cat_pred.threshold,
            "mean_ensemble_std": float(np.mean(y_std)), "max_ensemble_std": float(np.max(y_std)),
            "mean_member_smooth_l1": float(np.mean(losses)),
            "best_member_smooth_l1": float(np.min(losses)),
            "worst_member_smooth_l1": float(np.max(losses)),
        })
        print(f"[{i + 1:03d}/{len(test_set):03d}] {name}: SmoothL1={reg['smooth_l1']:.6f}, "
              f"Accuracy={cls.accuracy:.3f}, F1={cls.f1:.3f}")

    write_csv(out_dir / "per_sample_metrics.csv", rows)
    write_csv(out_dir / "per_model_test_losses.csv", [
        {"ensemble_member": k + 1, "checkpoint": str(path),
         "mean_test_smooth_l1": float(np.mean(l)), "std_test_smooth_l1": float(np.std(l))}
        for k, (path, l) in enumerate(zip(checkpoints, member_losses))])

    overall = metrics_from_counts(*map(int, totals))
    mean_loss = float(np.mean([r["smooth_l1"] for r in rows]))
    mean_member = float(np.mean([r["mean_member_smooth_l1"] for r in rows]))
    summary = {
        "region": args.region,
        "n_test_samples": len(rows),
        "n_ensemble_models": len(models),
        "positive_quantile": C.POSITIVE_QUANTILE,
        "mean_ensemble_smooth_l1": mean_loss,
        "std_ensemble_smooth_l1": float(np.std([r["smooth_l1"] for r in rows])),
        "mean_ensemble_mae": float(np.mean([r["mae"] for r in rows])),
        "mean_ensemble_rmse": float(np.mean([r["rmse"] for r in rows])),
        "mean_member_smooth_l1": mean_member,
        "ensemble_gain_over_mean_member_fraction": (mean_member - mean_loss) / mean_member,
        "mean_prediction_std_across_models": float(np.mean([r["mean_ensemble_std"] for r in rows])),
        "global_accuracy": overall.accuracy, "global_precision": overall.precision,
        "global_recall": overall.recall, "global_f1": overall.f1,
        "total_tp": overall.tp, "total_tn": overall.tn,
        "total_fp": overall.fp, "total_fn": overall.fn,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({**summary, "checkpoints": [str(p) for p in checkpoints]}, f, indent=2)
    write_csv(out_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
