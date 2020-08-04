r"""Residual block with RNN layers.

Usage:
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
    r"""Residual block with RNN layers.

    out = activate(F(x) + x)

    Args:
        d_in:
            Input data's vector dimension.
        d_out:
            GRU layers hidden dimension.
        dropout:
            Dropout probability on all layers out (except output layer).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float
    ):
        super().__init__()

        self.linear_layer = torch.nn.Linear(
            in_features=d_in,
            out_features=d_out
        )

        self.rnn_layer = torch.nn.RNN(
            input_size=d_in,
            hidden_size=d_out,
            num_layers=1,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.act_fn = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        ht, _ = self.rnn_layer(x)

        if x.size(-1) != ht.size(-1):
            x = self.linear_layer(x)

        return self.act_fn(self.dropout(ht + x))
