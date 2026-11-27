"""Sensitivity of the U-Net to the number of training years (Region A).

For every start year s = 0..14, the training set is the blocks [s, 28) minus
the two validation years, i.e. 26 - s years. For each training size, 10
models are trained and evaluated on the validation years, and their bagged
prediction is evaluated on the test period. The paper keeps s = 4 (22 years).

Writes to <output>/cv_training_years/:
    <s>__run/          per-size checkpoints, histories and statistics
    all_stats.npz      losses as a function of the training size
    losses_val_test.png

Usage:  python ml/01_training_size.py
"""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config as C
from data import VelocityDataset, test_indices, validation_indices, year_ids_to_indices
from model import UNet
from training import limit_threads, train_with_validation

N_REPEATS = 10
EPOCHS = 10
START_YEARS = range(15)


def train_one_repeat(k, train_idx, val_idx, run_dir):
    limit_threads(C.THREADS_PER_WORKER)
    dataset = VelocityDataset(C.REGION_FILES["A"], crop=C.CROP)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)

    model = UNet(in_channels=C.IN_CHANNELS, out_channels=C.OUT_CHANNELS, cropped=C.CROP)
    train_losses, val_losses = train_with_validation(
        model, train_loader, val_loader, lr=C.LEARNING_RATE, epochs=EPOCHS)

    out = run_dir / f"{k}__training"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"model_k{k}.pth")
    np.savez(out / f"history_k{k}.npz",
             train_losses=np.array(train_losses), val_losses=np.array(val_losses),
             last_val_loss=val_losses[-1], best_val_loss=np.min(val_losses),
             last_train_loss=train_losses[-1])
    return k, float(val_losses[-1]), float(np.min(val_losses)), float(train_losses[-1])


def evaluate_bagged(run_dir, test_idx):
    """Smooth-L1 test loss of every model and of their average."""
    criterion = nn.SmoothL1Loss()
    test_set = Subset(VelocityDataset(C.REGION_FILES["A"], crop=C.CROP), test_idx)

    models = []
    for k in range(N_REPEATS):
        model = UNet(in_channels=C.IN_CHANNELS, out_channels=C.OUT_CHANNELS, cropped=C.CROP)
        model.load_state_dict(torch.load(run_dir / f"{k}__training" / f"model_k{k}.pth",
                                         map_location="cpu"))
        models.append(model.eval())

    bagged, single = [], [[] for _ in models]
    with torch.no_grad():
        for x, y in test_set:
            y = y.squeeze(0)
            preds = [m(x.unsqueeze(0)).squeeze(0).squeeze(0) for m in models]
            for k, p in enumerate(preds):
                single[k].append(float(criterion(p, y).item()))
            bagged.append(float(criterion(torch.stack(preds).mean(dim=0), y).item()))

    per_model = np.mean(np.array(single), axis=1)
    return {
        "bagged_test_loss": float(np.mean(bagged)),
        "single_model_test_mean": float(np.mean(per_model)),
        "single_model_test_std": float(np.std(per_model, ddof=1)),
        "bagged_test_losses_per_sample": np.array(bagged),
        "single_model_test_losses": np.array(single),
    }


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    out_dir = C.CV_TRAINING_YEARS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    limit_threads(C.THREADS_PER_WORKER)

    n_samples = len(VelocityDataset(C.REGION_FILES["A"], crop=C.CROP))
    val_idx = validation_indices()
    test_idx = test_indices(n_samples)

    stats = {key: [] for key in (
        "training_size_list", "mean_last_val_losses", "std_last_val_losses",
        "mean_best_val_losses", "std_best_val_losses", "mean_last_train_losses",
        "bagged_test_losses", "single_model_test_means", "single_model_test_stds")}

    for start_year in START_YEARS:
        run_dir = out_dir / f"{start_year}__run"
        run_dir.mkdir(exist_ok=True)
        years = [y for y in range(start_year, C.TEST_START_YEAR) if y not in C.VALIDATION_YEARS]
        train_idx = year_ids_to_indices(years, n_samples)
        n_years = len(train_idx) / C.SAMPLES_PER_YEAR
        print(f"\n--- start year {start_year}: {len(train_idx)} training samples "
              f"({n_years:.2f} years) ---")

        last_val, best_val, last_train = (np.zeros(N_REPEATS) for _ in range(3))
        with ProcessPoolExecutor(max_workers=C.MAX_WORKERS) as pool:
            futures = [pool.submit(train_one_repeat, k, train_idx, val_idx, run_dir)
                       for k in range(N_REPEATS)]
            for future in as_completed(futures):
                k, last_val[k], best_val[k], last_train[k] = future.result()
                print(f"repeat {k}: last val = {last_val[k]:.6e}, best val = {best_val[k]:.6e}")

        np.savez(run_dir / f"{start_year}_val_stats.npz",
                 last_val_losses=last_val, best_val_losses=best_val,
                 last_train_losses=last_train,
                 mean_last_val_loss=last_val.mean(), std_last_val_loss=last_val.std(ddof=1),
                 mean_best_val_loss=best_val.mean(), std_best_val_loss=best_val.std(ddof=1),
                 mean_last_train_loss=last_train.mean())

        test = evaluate_bagged(run_dir, test_idx)
        np.savez(run_dir / f"{start_year}_test_stats.npz", **test)

        stats["training_size_list"].append(n_years)
        stats["mean_last_val_losses"].append(last_val.mean())
        stats["std_last_val_losses"].append(last_val.std(ddof=1))
        stats["mean_best_val_losses"].append(best_val.mean())
        stats["std_best_val_losses"].append(best_val.std(ddof=1))
        stats["mean_last_train_losses"].append(last_train.mean())
        stats["bagged_test_losses"].append(test["bagged_test_loss"])
        stats["single_model_test_means"].append(test["single_model_test_mean"])
        stats["single_model_test_stds"].append(test["single_model_test_std"])
        print(f"validation {last_val.mean():.6e} +/- {last_val.std(ddof=1):.6e} | "
              f"bagged test {test['bagged_test_loss']:.6e}")

    stats = {k: np.array(v) for k, v in stats.items()}
    np.savez(out_dir / "all_stats.npz", **stats)

    plt.figure()
    plt.errorbar(stats["training_size_list"], stats["mean_last_val_losses"],
                 yerr=stats["std_last_val_losses"], capsize=3, label="last validation loss")
    plt.plot(stats["training_size_list"], stats["bagged_test_losses"], "x-",
             label="bagged test loss")
    plt.errorbar(stats["training_size_list"], stats["single_model_test_means"],
                 yerr=stats["single_model_test_stds"], capsize=3, linestyle="--",
                 marker="o", label="single-model test loss")
    plt.xlabel("Number of years in training")
    plt.ylabel("SmoothL1 loss")
    plt.legend()
    plt.savefig(out_dir / "losses_val_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
