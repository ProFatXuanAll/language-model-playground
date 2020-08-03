r"""Language model with residual GRU blocks.

Usage:
    from torch.utils.data import DataLoader
    import lmp

    model = lmp.model.ResidualGRUModel(...)
    tokenizer = lmp.tokenizer.CharDictTokenizer(...)
    dataset = lmp.dataset.BaseDataset(...)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=dataset.create_collate_fn(tokenizer)
    )
    for x, y in dataloader:
        pred = model(x)
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

from lmp.model._base_residual_rnn_model import BaseResidualRNNModel


class ResidualGRUModel(BaseResidualRNNModel):
    r"""Language model with residual GRU blocks.

    Each input token will first be embedded into vectors, then sequentially
    feed into residual GRU blocks. Output vectors of blocks then go through
    fully-connected layer and project back to embedding dimension in order to
    perform vocabulary prediction.

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
            Dropout probability on all layers out (except output layer).
        num_rnn_layers:
            Number of RNN layers to use.
        num_linear_layers:
            Number of Linear layers to use.
        pad_token_id:
            Padding token's id. Embedding layers will initialize padding
            token's vector with zeros.
        vocab_size:
            Embedding matrix vocabulary dimension.
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

        # Sequential residual GRU blocks
        # Dimension: (E, H)
        gru_blocks = []
        for _ in range(num_rnn_layers):
            gru_blocks.append(
                lmp.model.ResGRUBlock(
                    d_in=d_hid,
                    d_out=d_hid,
                    dropout=dropout
                )
            )

        self.rnn_blocks = torch.nn.Sequential(*gru_blocks)
