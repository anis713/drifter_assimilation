"""Input distribution shift between training, Region A test and Region B (Fig. 10).

Each sample is represented by its first-day velocities [Ub, Vb, Ut, Vt]
(4 x 16 x 16 = 1024 features). A PCA fitted on the training samples keeps
95 % of their variance; in that space, the Mahalanobis distance to the
training distribution uses a Ledoit-Wolf covariance. D_95 is the 95th
percentile of the training distances, and f_95 is the fraction of samples
of each set beyond it.

Writes: <output>/figures/mahalanobis_distribution.pdf and prints f_95.

Usage:  python ml/fig_mahalanobis.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

import config as C
from data import load_records, test_indices, year_ids_to_indices

VARIANCE_KEPT = 0.95
OOD_PERCENTILE = 95


def first_day_inputs(records, indices):
    """(n, 1024) array of the first-day [Ub, Vb, Ut, Vt] of the given samples."""
    Ub = np.zeros((len(indices), 2, 16, 16))
    Ut = np.zeros((len(indices), 2, 16, 16))
    for n, i in enumerate(indices):
        Ub[n] = records[i]["Ub"][0], records[i]["Vb"][0]
        Ut[n] = records[i]["Ut"][0], records[i]["Vt"][0]
    return np.concatenate([Ub.reshape(len(Ub), -1), Ut.reshape(len(Ut), -1)], axis=1)


def mahalanobis(Z, mu, precision):
    diff = Z - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, precision, diff), 0))


def main():
    _, rec_a = load_records(C.REGION_FILES["A"])
    _, rec_b = load_records(C.REGION_FILES["B"])
    X_train = first_day_inputs(rec_a, year_ids_to_indices(C.TRAIN_YEAR_POOL, len(rec_a)))
    X_test = first_day_inputs(rec_a, test_indices(len(rec_a))[:C.DEPLOYMENT_SAMPLES["A"]])
    X_b = first_day_inputs(rec_b, test_indices(len(rec_b))[:C.DEPLOYMENT_SAMPLES["B"]])
    print(f"train {X_train.shape}, Region A test {X_test.shape}, Region B {X_b.shape}")

    cumulative = np.cumsum(PCA().fit(X_train).explained_variance_ratio_)
    n_components = np.searchsorted(cumulative, VARIANCE_KEPT) + 1
    pca = PCA(n_components=n_components, svd_solver="full")
    Z_train = pca.fit_transform(X_train)
    Z_test, Z_b = pca.transform(X_test), pca.transform(X_b)
    print(f"{n_components} principal components "
          f"({100 * pca.explained_variance_ratio_.sum():.2f} % of the variance)")

    cov = LedoitWolf().fit(Z_train)
    D_train = mahalanobis(Z_train, cov.location_, cov.precision_)
    D_test = mahalanobis(Z_test, cov.location_, cov.precision_)
    D_b = mahalanobis(Z_b, cov.location_, cov.precision_)

    threshold = np.percentile(D_train, OOD_PERCENTILE)
    print(f"D_95 = {threshold:.4f}")
    for name, D in (("training", D_train), ("Region A test", D_test), ("Region B", D_b)):
        print(f"  {name:14s} f_95 = {100 * np.mean(D > threshold):.2f} %, "
              f"median D = {np.median(D):.3f}")

    all_d = np.concatenate([D_train, D_test, D_b])
    bins = np.linspace(all_d.min(), all_d.max(), 30)
    plt.figure(figsize=(7, 5))
    plt.rcParams["hatch.linewidth"] = 1.5
    plt.hist(D_train, bins=bins, density=True, facecolor="none", edgecolor="#5D0D7D",
             linewidth=2.0, label="Region A training")
    plt.hist(D_test, bins=bins, density=True, facecolor="none", edgecolor="#d94747",
             linewidth=2.0, label="Region A test")
    plt.hist(D_b, bins=bins, density=True, color="#298c8c", linewidth=2.0, alpha=0.5,
             label="Region B")
    plt.axvline(threshold, color="k", linestyle="--", linewidth=2.5,
                label=rf"$D_{{95}} = {threshold:.02f}$")
    plt.xlabel(r"Mahalanobis distance $D_M$")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGURES_DIR / "mahalanobis_distribution.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {C.FIGURES_DIR / 'mahalanobis_distribution.pdf'}")


if __name__ == "__main__":
    main()
