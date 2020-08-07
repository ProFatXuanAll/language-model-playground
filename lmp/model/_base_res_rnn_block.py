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

    out = activate(F(x) + x)

    Args:
        d_hid:
            GRU layers hidden dimension.
        dropout:
            Dropout probability on all layers out (except output layer).
    """

    def __init__(
        self,
        d_hid: int,
        dropout: float
    ):
        super().__init__()

        self.rnn_layer = torch.nn.RNN(
            input_size=d_hid,
            hidden_size=d_hid,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.act_fn = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        ht, _ = self.rnn_layer(x)
        return self.act_fn(self.dropout(ht + x))
