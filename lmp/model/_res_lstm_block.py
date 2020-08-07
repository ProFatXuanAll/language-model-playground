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
        super().__init__(
            d_hid=d_hid,
            dropout=dropout
        )

        # Override RNN layer with LSTM layer.
        self.rnn_layer = torch.nn.LSTM(
            input_size=d_hid,
            hidden_size=d_hid,
            batch_first=True
        )
