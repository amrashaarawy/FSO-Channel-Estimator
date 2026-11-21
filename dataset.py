"""
dataset.py: windows (h_hat, h_t) into (past, future) training pairs and splits
the sequence in time order into train / val / test blocks.
"""
import torch
from torch.utils.data import Dataset


class ChannelDataset(Dataset):
    def __init__(self, h_hat, h_t_true, window, horizon):
        self.h_hat = torch.tensor(h_hat, dtype=torch.float32)
        self.h_t   = torch.tensor(h_t_true, dtype=torch.float32)
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return len(self.h_hat) - self.window - self.horizon + 1

    def __getitem__(self, i):
        x = self.h_hat[i : i + self.window].unsqueeze(-1)              # (window, 1)
        y = self.h_t[i + self.window + self.horizon - 1].unsqueeze(0)  # (1,)
        return x, y


def split_train_val_test(h_hat, h_t_true, train_frac, val_frac):
    """Time-ordered three-way split, no shuffle. Test is the remainder."""
    assert 0 < train_frac and 0 < val_frac and train_frac + val_frac < 1
    n = len(h_hat)
    i = int(n * train_frac)
    j = i + int(n * val_frac)
    return ((h_hat[:i], h_t_true[:i]),
            (h_hat[i:j], h_t_true[i:j]),
            (h_hat[j:], h_t_true[j:]))
