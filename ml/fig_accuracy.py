"""Classification accuracy of the bagged prediction over the test set (Fig. 4).

A cell is "positive" when its gain is at or above the median of the map
(clipped at zero). The accuracy of a sample is the fraction of cells whose
positive / not-positive label is predicted correctly. The samples shown in
Fig. 5 (test indices 0, 25 and 82) are marked (a)-(c).

Input:  <output>/evaluation_region_A/sample_*/ensemble_prediction.npz
Writes: <output>/figures/accuracy_plot_dates.{png,pdf}

Usage:  python ml/fig_accuracy.py
"""
import re
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import config as C
from ensemble import categorize, classification_metrics, metrics_from_counts

HIGHLIGHT = {0: ("(a)", (0, -8)), 25: ("(b)", (0, -8)), 82: ("(c)", (15, 5))}


def main():
    folders = sorted((p for p in C.evaluation_dir("A").iterdir() if p.is_dir()),
                     key=lambda p: int(re.search(r"sample_(\d+)", p.name).group(1)))

    dates, scores = [], []
    for folder in folders:
        data = np.load(folder / "ensemble_prediction.npz")
        cls = classification_metrics(categorize(data["y_true"], C.POSITIVE_QUANTILE).positive,
                                     categorize(data["y_pred_mean"], C.POSITIVE_QUANTILE).positive)
        scores.append(cls)
        dates.append(datetime.strptime(re.search(r"map_(\d{4}-\d{2}-\d{2})_to_", folder.name)
                                       .group(1), "%Y-%m-%d"))

    accuracy = np.array([s.accuracy for s in scores])
    dates = np.array(dates)
    # Accuracy of the mean confusion matrix
    mean_accuracy = metrics_from_counts(*(np.mean([getattr(s, k) for s in scores])
                                          for k in ("tp", "tn", "fp", "fn"))).accuracy

    print(f"{len(accuracy)} test samples, mean accuracy {100 * mean_accuracy:.2f} %, "
          f"std {100 * accuracy.std(ddof=1):.2f} %, range {100 * accuracy.min():.1f}-"
          f"{100 * accuracy.max():.1f} %, {100 * np.mean(accuracy >= mean_accuracy):.1f} % "
          f"of the samples at or above the mean")
    for i, (label, _) in HIGHLIGHT.items():
        print(f"  {label} sample {i} ({dates[i]:%Y-%m-%d}): {100 * accuracy[i]:.1f} %")

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dates, 100 * accuracy, "-o", color="b", linewidth=0.8, markersize=3, label="Accuracy")
    for n, (i, (label, offset)) in enumerate(HIGHLIGHT.items()):
        ax.plot(dates[i], 100 * accuracy[i], "o", color="r", markersize=6,
                label="Selected samples" if n == 0 else None)
        ax.annotate(label, (dates[i], 100 * accuracy[i]), xytext=offset,
                    textcoords="offset points", ha="center", va="top", fontsize=12)
    ax.axhline(100 * mean_accuracy, color="k", linestyle="--", linewidth=1.5,
               label=f"Mean accuracy = {100 * mean_accuracy:.2f}%")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Test sample starting date", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy variability across the test set", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=1, fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES_DIR / f"accuracy_plot_dates.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {C.FIGURES_DIR / 'accuracy_plot_dates.pdf'}")


if __name__ == "__main__":
    main()
