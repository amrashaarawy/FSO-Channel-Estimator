"""
models.py: the predictors to be compared (linear baseline now, LSTM added later).

contract every model must honour:
    input  x : (batch, window, 1) float32 -- noisy channel estimates
    output   : (batch, 1)                 -- predicted h_t, k steps past the window

"""
import torch.nn as nn

import config


class LinearPredictor(nn.Module):
    """baseline linear prediction"""

    def __init__(self, window):
        super().__init__()
        self.linear = nn.Linear(window, 1)    # one weight per past sample + a bias

    def forward(self, x):                    # x: (batch, window, 1)
        x = x.squeeze(-1)                   
        return self.linear(x)                  # (batch, 1)


def make_model(kind):
    """the one place that branches on model type"""
    if kind == "linear":
        return LinearPredictor(config.WINDOW_N)
    raise ValueError(f"unknown model kind: {kind!r}")