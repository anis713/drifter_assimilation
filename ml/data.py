"""Gain-map dataset, HDF5 access and the temporal train/validation/test split."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import SAMPLES_PER_YEAR, TEST_START_YEAR, VALIDATION_YEARS

GRID = 16   # every sample covers 16 x 16 grid points


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def group_names(h5_path: str | Path) -> list[str]:
    """Names of the samples, in file order (= chronological order)."""
    with h5py.File(h5_path, "r") as f:
        return [name for name in f if "io_map" in f[name]]


def load_record(h5_path: str | Path, group: str) -> dict[str, np.ndarray]:
    """All arrays of one sample: Ub, Vb, Ut, Vt (7, lon, lat), lon, lat, rdrmse."""
    with h5py.File(h5_path, "r") as f:
        io_map = f[group]["io_map"]
        return {key: io_map[key][()] for key in io_map}


def load_records(h5_path: str | Path) -> tuple[list[str], list[dict[str, np.ndarray]]]:
    """Names and arrays of every sample."""
    names, records = [], []
    with h5py.File(h5_path, "r") as f:
        for name in f:
            if "io_map" in f[name]:
                io_map = f[name]["io_map"]
                names.append(name)
                records.append({key: io_map[key][()] for key in io_map})
    return names, records


def test_sample(h5_path: str | Path, local_index: int) -> tuple[str, dict[str, np.ndarray]]:
    """Name and arrays of the ``local_index``-th test sample.

    The gain map is returned under ``"true_map"``, transposed exactly as the
    network target is (see :class:`VelocityDataset`).
    """
    names = group_names(h5_path)
    test_names = names[TEST_START_YEAR * SAMPLES_PER_YEAR:]
    name = test_names[local_index]
    record = load_record(h5_path, name)
    record["true_map"] = record["rdrmse"].T
    return name, record


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def year_ids_to_indices(year_ids, n_samples: int,
                        samples_per_year: int = SAMPLES_PER_YEAR) -> np.ndarray:
    """Sample indices of the given yearly blocks."""
    return np.concatenate([
        np.arange(y * samples_per_year, min((y + 1) * samples_per_year, n_samples))
        for y in year_ids
    ])


def validation_indices(samples_per_year: int = SAMPLES_PER_YEAR) -> np.ndarray:
    return np.concatenate([
        np.arange(y * samples_per_year, (y + 1) * samples_per_year)
        for y in VALIDATION_YEARS
    ])


def test_indices(n_samples: int, samples_per_year: int = SAMPLES_PER_YEAR) -> np.ndarray:
    start = TEST_START_YEAR * samples_per_year
    if n_samples <= start:
        raise ValueError(f"No test samples: dataset has {n_samples} samples, "
                         f"the test set starts at index {start}.")
    return np.arange(start, n_samples)


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class VelocityDataset(Dataset):
    """First-day velocities (input) and gain map (target) of every sample.

    X: (4, 16, 16) = standardised [Ub, Vb, Ut, Vt] on the first day.
    Y: (1, 16 - 2*crop, 16 - 2*crop) = standardised gain map, transposed
       from its storage order so that it is indexed [lon, lat] like the
       velocities.

    ``means``/``stds`` default to 0/1, i.e. no standardisation (as used for
    every model of the paper).
    """

    def __init__(self, h5_path, means=(0, 0, 0, 0, 0), stds=(1, 1, 1, 1, 1), crop=3):
        self.crop = crop
        _, self.records = load_records(h5_path)
        if not self.records:
            raise RuntimeError(f"No io_map groups found in {h5_path}")
        self.means = means      # ub, vb, ut, vt, map
        self.stds = stds

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        data = self.records[idx]
        m, s = self.means, self.stds

        channels = [
            (data[key][0].reshape(1, GRID, GRID) - m[i]) / s[i]
            for i, key in enumerate(("Ub", "Vb", "Ut", "Vt"))
        ]
        x = np.concatenate(channels, axis=0)
        y = (data["rdrmse"].T - m[4]) / s[4]

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(0))
