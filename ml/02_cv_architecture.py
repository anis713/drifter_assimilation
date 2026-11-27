"""Blocked 5-fold cross-validation of the U-Net architecture (Region A).

The 22 training years are split into 5 consecutive blocks; each block serves
once as validation. Each configuration is trained with 3 seeds per fold, with
early stopping on the validation loss.

Experiments:
    depth          3 vs 4 encoder levels: (64,128,256) and (64,128,256,512),
                   no dropout
    width_dropout  widths (16,32,64), (32,64,128), (64,128,256) crossed with
                   bottleneck dropout 0, 0.1 and 0.2

Writes to <output>/cv_depth/ or <output>/cv_width_dropout/:
    config_XXX/fold_YY/seed_ZZ/   best_model.pth, history.npz, result.json
    configs.json, folds.json, cv_raw_results.csv,
    cv_config_summary.{csv,json}, cv_all_stats.npz and summary figures

Usage:  python ml/02_cv_architecture.py --experiment depth
        python ml/02_cv_architecture.py --experiment width_dropout
"""
import argparse
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config as C
from data import VelocityDataset, year_ids_to_indices
from model import UNet
from training import limit_threads, set_seed, train_early_stopping

EXPERIMENTS = {
    "depth": [
        {"features": [64, 128, 256], "dropout": 0.0},
        {"features": [64, 128, 256, 512], "dropout": 0.0},
    ],
    "width_dropout": [
        {"features": features, "dropout": dropout}
        for features in ([16, 32, 64], [32, 64, 128], [64, 128, 256])
        for dropout in (0.0, 0.1, 0.2)
    ],
}


def make_folds():
    folds = []
    for fold_id, block in enumerate(np.array_split(np.array(C.TRAIN_YEAR_POOL), C.CV_N_FOLDS)):
        val_years = [int(y) for y in block]
        folds.append({
            "fold_id": fold_id,
            "train_years": [y for y in C.TRAIN_YEAR_POOL if y not in val_years],
            "val_years": val_years,
        })
    return folds


def run_task(task, n_samples, out_dir):
    limit_threads(C.THREADS_PER_WORKER)
    set_seed(task["seed"])

    dataset = VelocityDataset(C.REGION_FILES["A"], crop=C.CROP)
    train_set = Subset(dataset, year_ids_to_indices(task["train_years"], n_samples))
    val_set = Subset(dataset, year_ids_to_indices(task["val_years"], n_samples))
    train_loader = DataLoader(train_set, batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = UNet(in_channels=C.IN_CHANNELS, out_channels=C.OUT_CHANNELS, cropped=C.CROP,
                 features=tuple(task["features"]), dropout=task["dropout"],
                 dropout_mode="bottleneck")
    train_losses, val_losses, best_val, best_epoch = train_early_stopping(
        model, train_loader, val_loader, lr=C.LEARNING_RATE, max_epochs=C.CV_MAX_EPOCHS,
        patience=C.CV_PATIENCE, min_delta=C.CV_MIN_DELTA)

    task_dir = (out_dir / f"config_{task['config_id']:03d}" / f"fold_{task['fold_id']:02d}"
                / f"seed_{task['seed']:02d}")
    task_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), task_dir / "best_model.pth")
    np.savez(task_dir / "history.npz",
             train_losses=np.array(train_losses), val_losses=np.array(val_losses),
             best_val_loss=best_val, best_epoch=best_epoch,
             final_val_loss=val_losses[-1], final_train_loss=train_losses[-1],
             train_years=np.array(task["train_years"]), val_years=np.array(task["val_years"]),
             features=np.array(task["features"]), dropout=task["dropout"])

    result = {
        **{k: task[k] for k in ("config_id", "fold_id", "seed", "features", "dropout",
                                "train_years", "val_years")},
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "final_val_loss": float(val_losses[-1]),
        "final_train_loss": float(train_losses[-1]),
        "n_epochs_ran": len(val_losses),
    }
    with open(task_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def summarize(configs, results):
    summary = []
    for config in configs:
        rows = [r for r in results if r["config_id"] == config["config_id"]]
        vals = np.array([r["best_val_loss"] for r in rows])
        epochs = np.array([r["best_epoch"] for r in rows])
        summary.append({
            **config,
            "mean_best_val_loss": float(vals.mean()),
            "std_best_val_loss": float(vals.std(ddof=1)),
            "median_best_val_loss": float(np.median(vals)),
            "min_best_val_loss": float(vals.min()),
            "max_best_val_loss": float(vals.max()),
            "mean_best_epoch": float(epochs.mean()),
            "std_best_epoch": float(epochs.std(ddof=1)),
        })
    return sorted(summary, key=lambda s: s["mean_best_val_loss"])


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) if isinstance(v, list) else v for k, v in row.items()})


def plot_summary(summary, results, out_dir):
    labels = [f"{s['features']}, d={s['dropout']}" for s in summary]
    ranks = np.arange(1, len(summary) + 1)

    for key, ylabel, name in [
        ("best_val_loss", "Mean best validation loss", "cv_ranked_configs.png"),
        ("best_epoch", "Best epoch", "cv_best_epoch_by_config.png"),
    ]:
        plt.figure(figsize=(11, 5))
        plt.errorbar(ranks, [s[f"mean_{key}"] for s in summary],
                     yerr=[s[f"std_{key}"] for s in summary],
                     marker="o", capsize=3, linestyle="none")
        plt.xticks(ranks, labels, rotation=45, ha="right")
        plt.xlabel("Configuration rank")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=300, bbox_inches="tight")
        plt.close()

    # Fold-to-fold variability of the best configuration
    best = summary[0]["config_id"]
    means, stds = [], []
    for fold_id in range(C.CV_N_FOLDS):
        vals = np.array([r["best_val_loss"] for r in results
                         if r["config_id"] == best and r["fold_id"] == fold_id])
        means.append(vals.mean())
        stds.append(vals.std(ddof=1))
    plt.figure(figsize=(7, 5))
    plt.errorbar(np.arange(C.CV_N_FOLDS), means, yerr=stds, marker="o", capsize=3)
    plt.xlabel("Fold")
    plt.ylabel("Best validation loss")
    plt.title(f"Best configuration ({best}) across folds")
    plt.tight_layout()
    plt.savefig(out_dir / "cv_best_config_fold_losses.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dropout sensitivity by width (width_dropout experiment only)
    dropouts = sorted({s["dropout"] for s in summary})
    if len(dropouts) > 1:
        plt.figure(figsize=(8, 5))
        for features in {tuple(s["features"]) for s in summary}:
            rows = sorted((s for s in summary if tuple(s["features"]) == features),
                          key=lambda s: s["dropout"])
            plt.errorbar([s["dropout"] for s in rows], [s["mean_best_val_loss"] for s in rows],
                         yerr=[s["std_best_val_loss"] for s in rows], marker="o", capsize=3,
                         label=str(list(features)))
        plt.xlabel("Dropout")
        plt.ylabel("Mean best validation loss")
        plt.legend(title="features")
        plt.tight_layout()
        plt.savefig(out_dir / "cv_dropout_by_architecture.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    args = parser.parse_args()

    out_dir = C.CV_ARCHITECTURE_DIRS[args.experiment]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(VelocityDataset(C.REGION_FILES["A"], crop=C.CROP))
    configs = [{"config_id": i, **c} for i, c in enumerate(EXPERIMENTS[args.experiment])]
    folds = make_folds()
    tasks = [{**config, **fold, "seed": seed}
             for config in configs for fold in folds for seed in range(C.CV_N_SEEDS)]

    with open(out_dir / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)
    with open(out_dir / "folds.json", "w") as f:
        json.dump(folds, f, indent=2)
    print(f"{len(configs)} configurations x {len(folds)} folds x {C.CV_N_SEEDS} seeds "
          f"= {len(tasks)} trainings -> {out_dir}")

    results = []
    with ProcessPoolExecutor(max_workers=C.MAX_WORKERS) as pool:
        futures = [pool.submit(run_task, task, n_samples, out_dir) for task in tasks]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(f"config={r['config_id']:03d} fold={r['fold_id']} seed={r['seed']} "
                  f"best_val={r['best_val_loss']:.6e} best_epoch={r['best_epoch']}")

    write_csv(out_dir / "cv_raw_results.csv", results)
    summary = summarize(configs, results)
    write_csv(out_dir / "cv_config_summary.csv",
              [{"rank": i, **s} for i, s in enumerate(summary, start=1)])
    with open(out_dir / "cv_config_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(out_dir / "cv_all_stats.npz",
             **{key: np.array([s[src] for s in summary]) for key, src in [
                 ("config_ids", "config_id"),
                 ("mean_best_val_losses", "mean_best_val_loss"),
                 ("std_best_val_losses", "std_best_val_loss"),
                 ("median_best_val_losses", "median_best_val_loss"),
                 ("mean_best_epochs", "mean_best_epoch"),
                 ("std_best_epochs", "std_best_epoch"),
                 ("dropout", "dropout")]})
    plot_summary(summary, results, out_dir)

    print("\nBest configuration:")
    print(json.dumps(summary[0], indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
