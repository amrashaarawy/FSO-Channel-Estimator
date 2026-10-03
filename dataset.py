"""
dataset.py — forms the model's realistic input (h_hat) from r_e, and
windows (h_hat, h_t) into (past, future) training pairs.
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
        return len(self.h_hat) - self.window - self.horizon

    def __getitem__(self, i):
        x = self.h_hat[i : i + self.window].unsqueeze(-1)              # (window, 1)
        y = self.h_t[i + self.window + self.horizon - 1].unsqueeze(0)  # (1,)
        return x, y


def split_train_test(h_hat, h_t_true, train_frac):
    """Time-based split, no shuffle"""
    n = len(h_hat)
    split = int(n * train_frac)
    return (h_hat[:split], h_t_true[:split]), (h_hat[split:], h_t_true[split:])