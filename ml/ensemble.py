"""Loading the bagged ensemble, predicting, and classifying gain maps."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CROP, ENSEMBLE_DIR, N_ENSEMBLE
from model import UNet


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def natural_sort_key(value) -> list:
    """Sort key that puts model_2 before model_10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def ensemble_checkpoints(model_dir: Path = ENSEMBLE_DIR, n_members: int = N_ENSEMBLE) -> list[Path]:
    """The first ``n_members`` checkpoints (model_seed_00, model_seed_01, ...).

    The ensemble-size curve is cumulative, so the first 12 seeds are the
    12-member point of that curve.
    """
    checkpoints = sorted(Path(model_dir).glob("**/*.pth"), key=natural_sort_key)
    if len(checkpoints) < n_members:
        raise FileNotFoundError(
            f"Found {len(checkpoints)} checkpoints under {model_dir}, need {n_members}.")
    return checkpoints[:n_members]


def features_from_state_dict(state_dict) -> list[int]:
    """Encoder widths stored in a U-Net state dict."""
    levels = sorted(int(m.group(1)) for key in state_dict
                    if (m := re.match(r"^downs\.(\d+)\.net\.0\.weight$", key)))
    return [int(state_dict[f"downs.{level}.net.0.weight"].shape[0]) for level in levels]


def load_model(path: Path, crop: int = CROP, device="cpu") -> UNet:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    in_channels = int(state_dict["downs.0.net.0.weight"].shape[1])
    out_channels = int(state_dict["final_conv.weight"].shape[0])
    model = UNet(in_channels=in_channels, out_channels=out_channels, cropped=crop,
                 features=features_from_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_ensemble(paths, crop: int = CROP, device="cpu") -> list[UNet]:
    models = [load_model(p, crop, device) for p in paths]
    print(f"Loaded {len(models)} models with features {models[0].features}")
    return models


@torch.inference_mode()
def predict_ensemble(models, x: torch.Tensor, device="cpu"):
    """Mean, (population) standard deviation and stack of the member predictions.

    x: one sample (C, H, W). Returns arrays of shape (h, w), (h, w), (n, h, w).
    """
    batch = x.unsqueeze(0).to(device)
    stack = np.stack([m(batch).squeeze(0).squeeze(0).cpu().numpy() for m in models])
    return stack.mean(axis=0), stack.std(axis=0, ddof=0), stack


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    smooth_l1 = F.smooth_l1_loss(torch.as_tensor(y_pred, dtype=torch.float32),
                                 torch.as_tensor(y_true, dtype=torch.float32)).item()
    return {
        "smooth_l1": float(smooth_l1),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
    }


# ---------------------------------------------------------------------------
# Negative / neutral / positive classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Categories:
    negative: np.ndarray
    neutral: np.ndarray
    positive: np.ndarray
    threshold: float
    codes: np.ndarray        # 0 negative, 1 neutral, 2 positive


def categorize(values: np.ndarray, positive_quantile: float) -> Categories:
    """negative: < 0;  neutral: [0, g*);  positive: >= g*.

    g* is the ``positive_quantile`` quantile of the map, clipped at zero so a
    negative value can never be labelled positive.
    """
    threshold = max(0.0, float(np.quantile(values, positive_quantile)))
    negative = values < 0.0
    positive = values >= threshold
    neutral = ~negative & ~positive

    codes = np.ones(values.shape, dtype=np.int8)
    codes[negative] = 0
    codes[positive] = 2
    return Categories(negative, neutral, positive, threshold, codes)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def _ratio(num, den) -> float:
    return float(num / den) if den != 0 else 0.0


def metrics_from_counts(tp, tn, fp, fn) -> ClassificationMetrics:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return ClassificationMetrics(
        accuracy=_ratio(tp + tn, tp + tn + fp + fn),
        precision=precision,
        recall=recall,
        f1=_ratio(2.0 * precision * recall, precision + recall),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def classification_metrics(true_positive: np.ndarray, pred_positive: np.ndarray) -> ClassificationMetrics:
    """Scores of the positive class, cell by cell."""
    return metrics_from_counts(
        tp=int(np.sum(true_positive & pred_positive)),
        tn=int(np.sum(~true_positive & ~pred_positive)),
        fp=int(np.sum(~true_positive & pred_positive)),
        fn=int(np.sum(true_positive & ~pred_positive)),
    )
