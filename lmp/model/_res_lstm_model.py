r"""Language model with residual LSTM blocks.

Usage:
    import lmp

    model = lmp.model.ResLSTMModel(...)
    logits = model(...)
    pred = model.predict(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# 3rd-party modules

import torch

# self-made modules

import lmp.model

from lmp.model._base_res_rnn_model import BaseResRNNModel


class ResLSTMModel(BaseResRNNModel):
    r"""Language model with residual LSTM blocks.

    Each input token will first be embedded into vectors, then project to
    hidden dimension. We then sequentially feed vectors into residual LSTM
    layer(s). Output vectors of residual LSTM layer(s) then go through
    fully-connected layer(s) and project back to embedding dimension in order
    to perform vocabulary prediction.

    In the comment below, we use following symbols to denote the size of
    each tensors:
        B: batch size
        S: sequence length
        E: embedding dimension
        V: vocabulary size
        H: hidden dimension

    Args:
        d_emb:
            Embedding matrix vector dimension.
        d_hid:
            RNN layers hidden dimension.
        dropout:
            Dropout probability on all layers output (except output layer).
        num_rnn_layers:
            Number of residual LSTM layers to use.
        num_linear_layers:
            Number of Linear layers to use.
        pad_token_id:
            Padding token's id. Embedding layers will initialize padding
            token's vector with zeros.
        vocab_size:
            Embedding matrix vocabulary dimension.

    Raises:
        TypeError:
            When one of the arguments are not instance of their type annotation
            respectively.
        ValueError:
            When one of the arguments do not follow their constraints. See
            docstring for arguments constraints.
    """

    def __init__(
            self,
            d_emb: int,
            d_hid: int,
            dropout: float,
            num_rnn_layers: int,
            num_linear_layers: int,
            pad_token_id: int,
            vocab_size: int
    ):
        super().__init__(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

        # Sequential residual LSTM blocks
        # Dimension: (E, E)
        rnn_blocks = []
        for _ in range(num_rnn_layers):
            rnn_blocks.append(
                lmp.model.ResLSTMBlock(
                    d_hid=d_hid,
                    dropout=dropout
                )
            )
        self.rnn_layer = torch.nn.Sequential(*rnn_blocks)
