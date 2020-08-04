r"""Residual LSTM block.

Usage:
    block = lmp.model.ResLSTMBlock(...)
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

# self-made modules

from lmp.model._base_res_rnn_block import BaseResRNNBlock


class ResLSTMBlock(BaseResRNNBlock):
    r"""Residual block with LSTM layers.

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
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout
        )

        # Override RNN layer with LSTM layer.
        self.rnn_layer = torch.nn.LSTM(
            input_size=d_in,
            hidden_size=d_out,
            num_layers=1,
            dropout=dropout,
            batch_first=True
        )
