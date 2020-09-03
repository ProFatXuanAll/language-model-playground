r"""Language model with self-attention GRU layers.

Usage:
    import lmp

    model = lmp.model.SelfAttentionResGRUModel(...)
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
import torch.nn

# self-made modules

from lmp.model._base_self_attention_res_rnn_model import BaseSelfAttentionResRNNModel
from lmp.model._res_gru_block import ResGRUBlock


class SelfAttentionResGRUModel(BaseSelfAttentionResRNNModel):
    r"""Language model with self-attention GRU layers.

    Each input token will first be embedded into vectors, then project to
    hidden dimension. We then sequentially feed vectors into RNN layer(s).
    Output vectors of RNN layer(s) then go through fully-connected layer(s) and
    project back to embedding dimension in order to perform vocabulary
    prediction.

    In the comment below, we use following symbols to denote the size of
    each tensors:
        B: Batch size.
        S: Sequence length.
        E: Embedding dimension.
        V: Vocabulary size.
        H: Hidden dimension.

    Args:
        d_emb:
            Embedding matrix vector dimension. Must be bigger than or equal to
            `1`.
        d_hid:
            RNN layers hidden dimension. Must be bigger than or equal to `1`.
        dropout:
            Dropout probability on all layers output (except output layer).
            Must range from `0.0` to `1.0`.
        num_linear_layers:
            Number of Linear layers to use. Must be bigger than or equal to
            `1`.
        num_rnn_layers:
            Number of RNN layers to use. Must be bigger than or equal to `1`.
        pad_token_id:
            Padding token's id. Embedding layer will initialize padding
            token's vector with zeros. Must be bigger than or equal to `0`, and
            must be smaller than `vocab_size`.
        vocab_size:
            Embedding matrix vocabulary dimension. Must be bigger than or equal
            to `1`.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type annotation
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
            num_linear_layers: int,
            num_rnn_layers: int,
            pad_token_id: int,
            vocab_size: int
    ):
        super().__init__(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_linear_layers=num_linear_layers,
            num_rnn_layers=num_rnn_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

        # Override residual RNN layer(s) with residual GRU layer(s).
        rnn_layer = []
        for _ in range(num_rnn_layers):
            rnn_layer.append(ResGRUBlock(d_hid=d_hid, dropout=dropout))
        self.rnn_layer = torch.nn.Sequential(*rnn_layer)
