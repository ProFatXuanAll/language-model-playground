r"""GRU residual block.

Usage:
    import lmp

    block = lmp.model.ResGRUBlock(...)
    logits = block(...)
"""


import torch
import torch.nn

from lmp.model._base_res_rnn_block import BaseResRNNBlock


class ResGRUBlock(BaseResRNNBlock):
    r"""GRU residual block.

    out = dropout(ReLU(GRU(x))) + x

    Args:
        d_hid:
            Residual GRU layer hidden dimension.
        dropout:
            Dropout probability on residual GRU layer output.

    TypeError:
            When `d_hid` is not an instance of `int` or `dropout` is not an
            instance of `float`.
        ValueError:
            When `d_hid < 1` or `dropout < 0` or `dropout > 1`.
    """

    def __init__(self, d_hid: int, dropout: float):
        super().__init__(d_hid=d_hid, dropout=dropout)

        # Override residual RNN block(s) with residual LSTM block(s).
        self.rnn_layer = torch.nn.GRU(
            input_size=d_hid,
            hidden_size=d_hid,
            batch_first=True
        )
