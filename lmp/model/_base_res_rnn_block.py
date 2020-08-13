r"""RNN residual block.

Usage:
    import lmp

    block = lmp.model.BaseResRNNBlock(...)
    logits = block(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd-party modules

import torch
import torch.nn


class BaseResRNNBlock(torch.nn.Module):
    r"""RNN residual block.

    out = x + dropout(activate(RNN(x)))

    Args:
        d_hid:
            GRU layers hidden dimension.
        dropout:
            Dropout probability on all layers out (except output layer).

    Raises:
        TypeError:
            When `d_hid` is not instance of `int` or `dropout` is not instance
            of `float`.
        ValueError:
            When `d_hid` is smaller than 1 or `dropout` is out range from `0.0`
            to `1.0`.
    """

    def __init__(
        self,
        d_hid: int,
        dropout: float
    ):
        super().__init__()
        # Type check.
        if not isinstance(d_hid, int):
            raise TypeError('`d_hid` must be instance of `int`.')

        if not isinstance(dropout, float):
            raise TypeError('`dropout` must be instance of `float`.')

        # Value Check.
        if d_hid < 1:
            raise ValueError('`d_hid` must be bigger than or equal to `1`.')

        if not (0 <= dropout <= 1):
            raise ValueError('`dropout` must range from `0.0` to `1.0`.')

        self.rnn_layer = torch.nn.RNN(
            input_size=d_hid,
            hidden_size=d_hid,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.act_fn = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ht, _ = self.rnn_layer(x)
        return x + self.dropout(self.act_fn(ht))
