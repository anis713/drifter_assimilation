"""Train the ensemble members and choose the ensemble size (Region A).

Trains N_TRAINED_MODELS (20) U-Nets with the selected architecture on the
22-year training pool, for FINAL_EPOCHS (26) epochs each, differing only in
their seed. Then evaluates the bagged prediction of the first M members,
M = 1..20, on the validation years. The paper keeps M = 12.

Writes to <output>/ensemble/:
    model_seed_XX/   model.pth, training_history.npz, result.json
    experiment_B_config.json, training_results.{json,csv}
    experiment_B_ensemble_stats.{npz,csv}, experiment_B_summary.txt
    ensemble_size_vs_val_loss.png, ensemble_relative_improvement.png,
    ensemble_marginal_gain.png, individual_model_val_losses.png

Usage:  python ml/03_train_ensemble.py              # train, then evaluate
        python ml/03_train_ensemble.py --skip-training
"""
import argparse
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config as C
from data import VelocityDataset, validation_indices, year_ids_to_indices
from model import UNet
from training import limit_threads, set_seed, train_fixed_epochs


def new_model():
    return UNet(in_channels=C.IN_CHANNELS, out_channels=C.OUT_CHANNELS, cropped=C.CROP,
                features=C.FEATURES, dropout=C.DROPOUT)


def train_one_model(seed, train_idx):
    limit_threads(C.THREADS_PER_WORKER)
    set_seed(seed)

    dataset = VelocityDataset(C.REGION_FILES["A"], crop=C.CROP)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=C.BATCH_SIZE, shuffle=True)
    model = new_model()
    losses = train_fixed_epochs(model, loader, lr=C.LEARNING_RATE, epochs=C.FINAL_EPOCHS)

    model_dir = C.ENSEMBLE_DIR / f"model_seed_{seed:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    np.savez(model_dir / "training_history.npz", train_losses=np.array(losses),
             final_train_loss=losses[-1], seed=seed, features=np.array(C.FEATURES),
             dropout=C.DROPOUT, final_epochs=C.FINAL_EPOCHS)
    result = {"seed": seed, "final_train_loss": float(losses[-1]),
              "features": list(C.FEATURES), "dropout": C.DROPOUT,
              "final_epochs": C.FINAL_EPOCHS}
    with open(model_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def ensemble_size_curve(eval_idx):
    """Smooth-L1 loss of the mean of the first M members, M = 1..N."""
    criterion = nn.SmoothL1Loss()
    eval_set = Subset(VelocityDataset(C.REGION_FILES["A"], crop=C.CROP), eval_idx)

    models = []
    for seed in range(C.N_TRAINED_MODELS):
        model = new_model()
        model.load_state_dict(torch.load(C.ENSEMBLE_DIR / f"model_seed_{seed:02d}" / "model.pth",
                                         map_location="cpu"))
        models.append(model.eval())

    n = len(models)
    per_model = [[] for _ in range(n)]
    per_size = {m: [] for m in range(1, n + 1)}
    with torch.no_grad():
        for x, y in eval_set:
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            preds = torch.stack([model(x) for model in models])
            for k in range(n):
                per_model[k].append(float(criterion(preds[k], y).item()))
            for m in range(1, n + 1):
                per_size[m].append(float(criterion(preds[:m].mean(dim=0), y).item()))

    mean_losses = np.array([np.mean(per_size[m]) for m in range(1, n + 1)])
    marginal_gain = np.zeros(n)
    marginal_gain[1:] = mean_losses[:-1] - mean_losses[1:]
    per_model = np.array(per_model)
    return {
        "ensemble_sizes": np.arange(1, n + 1),
        "mean_ensemble_losses": mean_losses,
        "std_ensemble_losses": np.array([np.std(per_size[m], ddof=1) for m in range(1, n + 1)]),
        "relative_improvement": 100 * (mean_losses[0] - mean_losses) / mean_losses[0],
        "marginal_gain": marginal_gain,
        "per_model_test_losses": per_model,
        "mean_per_model_test_losses": per_model.mean(axis=1),
    }


def plot_curve(stats, out_dir):
    sizes = stats["ensemble_sizes"]
    for y, ylabel, name, x in [
        (stats["mean_ensemble_losses"], "Bagged validation SmoothL1 loss",
         "ensemble_size_vs_val_loss.png", sizes),
        (stats["relative_improvement"], "Relative improvement over M=1 (%)",
         "ensemble_relative_improvement.png", sizes),
        (stats["marginal_gain"][1:], "Loss reduction from previous ensemble size",
         "ensemble_marginal_gain.png", sizes[1:]),
    ]:
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, "o-")
        if name != "ensemble_size_vs_val_loss.png":
            plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Ensemble size")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=300, bbox_inches="tight")
        plt.close()

    losses = stats["mean_per_model_test_losses"]
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(losses)), losses, "o-")
    plt.axhline(losses.mean(), linestyle="--", linewidth=1)
    plt.xlabel("Seed")
    plt.ylabel("Individual model validation loss")
    plt.tight_layout()
    plt.savefig(out_dir / "individual_model_val_losses.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-training", action="store_true",
                        help="only evaluate the ensemble-size curve")
    args = parser.parse_args()

    out_dir = C.ENSEMBLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(VelocityDataset(C.REGION_FILES["A"], crop=C.CROP))
    train_idx = year_ids_to_indices(C.TRAIN_YEAR_POOL, n_samples)
    val_idx = validation_indices()

    if not args.skip_training:
        config = {"features": list(C.FEATURES), "dropout": C.DROPOUT, "lr": C.LEARNING_RATE,
                  "batch_size": C.BATCH_SIZE, "final_epochs": C.FINAL_EPOCHS,
                  "n_models": C.N_TRAINED_MODELS, "train_year_pool": list(C.TRAIN_YEAR_POOL),
                  "n_train_samples": int(len(train_idx))}
        with open(out_dir / "experiment_B_config.json", "w") as f:
            json.dump(config, f, indent=2)

        results = []
        with ProcessPoolExecutor(max_workers=C.MAX_WORKERS) as pool:
            futures = [pool.submit(train_one_model, seed, train_idx)
                       for seed in range(C.N_TRAINED_MODELS)]
            for future in as_completed(futures):
                results.append(future.result())
                print(f"seed {results[-1]['seed']:02d}: final train loss "
                      f"{results[-1]['final_train_loss']:.6e}")
        results.sort(key=lambda r: r["seed"])
        with open(out_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(out_dir / "training_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "final_train_loss", "final_epochs"])
            writer.writerows([r["seed"], r["final_train_loss"], r["final_epochs"]] for r in results)

    stats = ensemble_size_curve(val_idx)
    np.savez(out_dir / "experiment_B_ensemble_stats.npz", **stats,
             train_idx=train_idx, test_idx=val_idx)
    with open(out_dir / "experiment_B_ensemble_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ensemble_size", "mean_val_loss", "std_val_loss_over_samples",
                         "relative_improvement_percent", "marginal_gain"])
        writer.writerows(zip(stats["ensemble_sizes"], stats["mean_ensemble_losses"],
                             stats["std_ensemble_losses"], stats["relative_improvement"],
                             stats["marginal_gain"]))

    individual = stats["mean_per_model_test_losses"]
    summary = (
        f"Ensemble size experiment (validation years, {len(val_idx)} samples)\n"
        f"features = {list(C.FEATURES)}, dropout = {C.DROPOUT}, epochs = {C.FINAL_EPOCHS}\n"
        f"M=1 loss  = {stats['mean_ensemble_losses'][0]:.8f}\n"
        f"M={C.N_TRAINED_MODELS} loss = {stats['mean_ensemble_losses'][-1]:.8f} "
        f"({stats['relative_improvement'][-1]:.4f} % better)\n"
        f"individual models: mean {individual.mean():.8f}, std {individual.std(ddof=1):.8f}, "
        f"best {individual.min():.8f}, worst {individual.max():.8f}\n"
    )
    print(summary)
    (out_dir / "experiment_B_summary.txt").write_text(summary)
    plot_curve(stats, out_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
