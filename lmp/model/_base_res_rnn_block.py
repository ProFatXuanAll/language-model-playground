r"""RNN residual block.

Usage:
    import lmp

    block = lmp.model.BaseResRNNBlock(...)
    logits = block(...)
"""


import torch
import torch.nn


class BaseResRNNBlock(torch.nn.Module):
    r"""RNN residual block.

    out = dropout(ReLU(RNN(x))) + x

    Args:
        d_hid:
            Residual RNN layer hidden dimension.
        dropout:
            Dropout probability on residual RNN layer output.

    Raises:
        TypeError:
            When `d_hid` is not an instance of `int` or `dropout` is not an
            instance of `float`.
        ValueError:
            When `d_hid < 1` or `dropout < 0` or `dropout > 1`.
    """

    def __init__(self, d_hid: int, dropout: float):
        super().__init__()

        # Type check.
        if not isinstance(d_hid, int):
            raise TypeError('`d_hid` must be an instance of `int`.')

        if not isinstance(dropout, float):
            raise TypeError('`dropout` must be an instance of `float`.')

        # Value Check.
        if d_hid < 1:
            raise ValueError('`d_hid` must be bigger than or equal to `1`.')

        if not 0 <= dropout <= 1:
            raise ValueError('`dropout` must range from `0.0` to `1.0`.')

        self.rnn_layer = torch.nn.RNN(
            input_size=d_hid,
            hidden_size=d_hid,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.act_fn = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Args:
            x:
                Batch of hidden vectors with numeric type `torch.float32`.

        Returns:
            Residual blocks output tensors.
        """
        ht, _ = self.rnn_layer(x)
        return self.dropout(self.act_fn(ht)) + x
