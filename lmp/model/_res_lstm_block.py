r"""LSTM residual block.

Usage:
    import lmp

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
    r"""LSTM residual block.

    out = dropout(activate(LSTM(x))) + x

    Args:
        d_hid:
            Residual LSTM layer hidden dimension.
        dropout:
            Dropout probability on residual LSTM layer output.

    TypeError:
            When `d_hid` is not an instance of `int` or `dropout` is not an
            instance of `float`.
        ValueError:
            When `d_hid < 1` or `dropout < 0` or `dropout > 1`.
    """

    def __init__(self, d_hid: int, dropout: float):
        super().__init__(d_hid=d_hid, dropout=dropout)

        # Override residual RNN block with residual LSTM block.
        self.rnn_layer = torch.nn.LSTM(
            input_size=d_hid,
            hidden_size=d_hid,
            batch_first=True
        )
