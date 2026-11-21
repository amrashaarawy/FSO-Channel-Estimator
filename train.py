"""
train.py: model-agnostic training loop. Knows nothing about which model it trains;
it only calls model(x) and model.parameters().
"""
import torch
import torch.nn as nn

import config


def mean_loss(model, loader, loss_fn):
    """average loss per sample over a loader, no learning"""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            total += loss.item() * len(xb)
            count += len(xb)
    return total / count


def train(model, train_loader, val_loader, epochs=config.NUM_EPOCHS, lr=config.LR):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
        history["train"].append(total / count)
        history["val"].append(mean_loss(model, val_loader, loss_fn))

    return history
