"""Training loops shared by the cross-validation and ensemble scripts.

All training uses the smooth-L1 loss and Adam, on the CPU.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def limit_threads(n_threads: int) -> None:
    """Keep each worker process on ``n_threads`` CPU threads."""
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)


def mean_batch_loss(model, loader, criterion) -> float:
    """Mean over batches of the loss (model in eval mode, no gradients)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            losses.append(float(criterion(model(x), y).item()))
    return float(np.mean(losses))


def train_early_stopping(model, train_loader, val_loader, lr, max_epochs,
                         patience, min_delta):
    """Train until the validation loss stops improving.

    Returns (train_losses, val_losses, best_val_loss, best_epoch) and leaves
    the weights of the best epoch in ``model``.
    """
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss, best_epoch, best_state = np.inf, -1, None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(mean_batch_loss(model, val_loader, criterion))

        if val_losses[-1] < best_val_loss - min_delta:
            best_val_loss, best_epoch = val_losses[-1], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses, best_val_loss, best_epoch


def train_fixed_epochs(model, train_loader, lr, epochs) -> list[float]:
    """Train for a fixed number of epochs; returns the mean batch loss per epoch."""
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    model.train()
    for _ in range(epochs):
        batch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(batch_losses)))
    return train_losses


def train_with_validation(model, train_loader, val_loader, lr, epochs):
    """Fixed-epoch training that also tracks the validation loss.

    Losses are sample-weighted means over each epoch. Used by the
    training-size experiment.
    """
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_losses.append(running / len(train_loader.dataset))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                running += criterion(model(x), y).item() * x.size(0)
        val_losses.append(running / len(val_loader.dataset))

        print(f"Epoch {epoch:03d} | Train {train_losses[-1]:.6f} | Val {val_losses[-1]:.6f}")

    return train_losses, val_losses
