"""Paths, data split and hyperparameters shared by every ML script.

All locations can be moved with two environment variables:

    DRIFTERS_ML_DATA     folder holding region_A.h5 and region_B.h5
                         (default: <repo>/data/ml)
    DRIFTERS_ML_OUTPUT   folder where checkpoints, predictions, deployment
                         results and figures are written
                         (default: <repo>/output/ml)

Importing this module also puts <repo>/src on ``sys.path`` so the scripts can
use the assimilation code (analysis_nDrif, sliding_window_assim, ...) and the
manifold code (manifolds.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DRIFTERS_ML_DATA", REPO_ROOT / "data" / "ml"))
OUTPUT_DIR = Path(os.environ.get("DRIFTERS_ML_OUTPUT", REPO_ROOT / "output" / "ml"))

# Gain-map datasets: one HDF5 group "map_<start>_to_<end>/io_map" per sample,
# holding Ub, Vb, Ut, Vt (7, 16, 16), lon (16,), lat (16,) and rdrmse (16-2*CROP)^2.
REGION_FILES = {
    "A": DATA_DIR / "region_A.h5",   # training / validation / test region
    "B": DATA_DIR / "region_B.h5",   # generalisation region (no retraining);
                                     # inputs only, its rdrmse fields are zero
}
# Files holding the true gain maps of the test samples. For Region B these
# were computed separately, for 160 test samples.
TRUTH_FILES = {
    "A": REGION_FILES["A"],
    "B": DATA_DIR / "region_B_test_truth.h5",
}

CV_TRAINING_YEARS_DIR = OUTPUT_DIR / "cv_training_years"
CV_ARCHITECTURE_DIRS = {
    "width_dropout": OUTPUT_DIR / "cv_width_dropout",
    "depth": OUTPUT_DIR / "cv_depth",
}
ENSEMBLE_DIR = OUTPUT_DIR / "ensemble"            # model_seed_XX/model.pth
FIGURES_DIR = OUTPUT_DIR / "figures"


def evaluation_dir(region: str) -> Path:
    """Per-sample ensemble predictions for one region."""
    return OUTPUT_DIR / f"evaluation_region_{region}"


def deployment_dir(region: str) -> Path:
    """Per-sample drifter-release experiments for one region."""
    return OUTPUT_DIR / f"deployment_region_{region}"


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
# Samples are weekly, so one "year" block holds 53 consecutive samples.
SAMPLES_PER_YEAR = len(range(0, 365, 7))           # 53

VALIDATION_YEARS = (14, 18)                         # blocks starting Jan 2008 and Apr 2012
TEST_START_YEAR = 28                                # test = every sample from block 28 on
                                                    # (Region A: 11 Nov 2022 - 31 Jul 2025)
# 22-year training pool selected by the training-size experiment
# (blocks 4-27 without the two validation blocks).
TRAIN_YEAR_POOL = tuple(
    [y for y in range(4, TEST_START_YEAR) if y not in VALIDATION_YEARS]
)

# ---------------------------------------------------------------------------
# Network and training
# ---------------------------------------------------------------------------
CROP = 3                       # cells removed on each side of the 16x16 input
IN_CHANNELS = 4                # Ub, Vb, Ut, Vt on the first day
OUT_CHANNELS = 1               # gain map
FEATURES = (64, 128, 256, 512)  # four encoder blocks, bottleneck 512 -> 1024
DROPOUT = 0.0
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
FINAL_EPOCHS = 26              # epochs used for the final ensemble members
N_TRAINED_MODELS = 20          # members trained for the ensemble-size curve
N_ENSEMBLE = 12                # members averaged in the final (bagged) model

# Early stopping used in the cross-validation experiments
CV_MAX_EPOCHS = 80
CV_PATIENCE = 10
CV_MIN_DELTA = 1e-4
CV_N_FOLDS = 5
CV_N_SEEDS = 3

# Parallel CPU training
MAX_WORKERS = int(os.environ.get("DRIFTERS_ML_WORKERS", 6))
THREADS_PER_WORKER = 1

# ---------------------------------------------------------------------------
# Gain-map classification
# ---------------------------------------------------------------------------
POSITIVE_QUANTILE = 0.50       # threshold between "neutral" and "positive" cells

# ---------------------------------------------------------------------------
# Drifter-release experiments (assimilation parameters of the paper)
# ---------------------------------------------------------------------------
ASSIMILATION = dict(
    nDaysSim=6,                # assimilation window (days)
    lfil=15000,                # filter length scale (m)
    dtAdvection=3600.0 / 2,
    dtObservation=3600.0 / 2,
    mylr=0.001,
    alpha=0.001,
    beta=0.1,
    n_drif=1,
    Tw_days=2,
    sigma_days=1,
    toll=1.0,
)
N_RANDOM_RELEASES = 50         # independent releases per test sample
# Number of test samples (taken from the start of the test period) used in
# the release experiments and in the distribution-shift analysis.
DEPLOYMENT_SAMPLES = {"A": 132, "B": 160}
# Reference true gain of the random-release experiment (green curve and
# denominator of mu_t in 06_summarize_deployments.py):
#   "positive_region": mean true gain over the CNN-selected release region
#                      (mean_pos_imp_truth, saved by 05)
#   "map_mean":        mean of the whole true gain map (from TRUTH_FILES)
RANDOM_REFERENCE = {"A": "positive_region", "B": "map_mean"}
MANIFOLD_OFFSET = 0.1          # half-width (deg) of the manifold neighbourhood
RANDOM_SEED = 47               # seeds the random releases of each test sample
STEPS_PER_DAY = 48             # RMSE time series sampling (dtAdvection = 30 min)
