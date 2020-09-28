r"""Language model with self-attention RNN layers.

Usage:
    import lmp

    model = lmp.model.BaseSelfAttentionRNNModel(...)
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

from lmp.model._attention_mechanism import attention_mechanism
from lmp.model._base_rnn_model import BaseRNNModel


class BaseSelfAttentionRNNModel(BaseRNNModel):
    r"""Language model with self-attention RNN layers.

    Each input token will first be embedded into vectors, then project to
    hidden dimension. We then sequentially feed vectors into RNN layer(s).
    And we get query, key, value vector by projecting output vectors of RNN
    layer(s).Passing query, key, value vector to do self-attention. Output
    vectors of self-attention then go through fully-connected layer(s) and
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

        self.proj_query = torch.nn.Linear(
            in_features=d_hid,
            out_features=d_hid
        )
        self.proj_key = torch.nn.Linear(
            in_features=d_hid,
            out_features=d_hid
        )
        self.proj_value = torch.nn.Linear(
            in_features=d_hid,
            out_features=d_hid
        )

        self.att_dropout = torch.nn.Dropout(dropout)

    def forward(self, batch_sequences: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Args:
            batch_sequences:
                Batch of sequences which have been encoded by
                `lmp.tokenizer.BaseTokenizer` with numeric type `torch.int64`.

        Returns:
            Logits for each token in sequences with numeric type `torch.float32`.
        """
        # 將 batch_sequences 中的所有 token_id 經過 embedding matrix
        # 轉換成 embedding vectors (共有 (B, S) 個維度為 E 的向量)
        # embedding 前的 batch_sequences 維度: (B, S)
        # embedding 後的 batch_sequences 維度: (B, S, E)
        batch_sequences = self.emb_dropout(self.emb_layer(batch_sequences))

        # 將每個 embedding vectors 經由 linear 轉換得到輸出 hidden vectors
        # ht 維度: (B, S, H)
        ht = self.proj_emb_to_hid(batch_sequences)

        # 將每個 embedding vectors 依序輸入 RNN 得到輸出 hidden vectors
        # ht 維度: (B, S, H)
        ht, _ = self.rnn_layer(ht)

        query = self.proj_query(ht)
        key = self.proj_key(ht)
        value = self.proj_value(ht)

        # 經過 attention 機制，進行 self-attention
        # ht 維度: (B, S, H)
        ht = self.att_dropout(attention_mechanism(query, key, value))

        # 將每個 hidden vectors 轉換維度至 embedding dimension
        # ht 維度: (B, S, E)
        ht = self.proj_hid_to_emb(ht)

        # 與轉置後的 embedding matrix 進行矩陣乘法取得預測文字
        # 重複使用 embedding matrix 的目的為節省參數數量
        # return 維度: (B, S, V)
        return ht.matmul(self.emb_layer.weight.transpose(0, 1))
