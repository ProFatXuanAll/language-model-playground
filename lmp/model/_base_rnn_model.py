r"""Language model with RNN layers.

Usage:
    import lmp

    model = lmp.model.BaseRNNModel(...)
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


class BaseRNNModel(torch.nn.Module):
    r"""Language model with pure RNN layers.

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
        super().__init__()

        # Type check.
        if not isinstance(d_emb, int):
            raise TypeError('`d_emb` must be an instance of `int`.')

        if not isinstance(d_hid, int):
            raise TypeError('`d_hid` must be an instance of `int`.')

        if not isinstance(dropout, float):
            raise TypeError('`dropout` must be an instance of `float`.')

        if not isinstance(num_linear_layers, int):
            raise TypeError(
                '`num_linear_layers` must be an instance of `int`.'
            )

        if not isinstance(num_rnn_layers, int):
            raise TypeError('`num_rnn_layers` must be an instance of `int`.')

        if not isinstance(pad_token_id, int):
            raise TypeError('`pad_token_id` must be an instance of `int`.')

        if not isinstance(vocab_size, int):
            raise TypeError('`vocab_size` must be an instance of `int`.')

        # Value Check.
        if d_emb < 1:
            raise ValueError('`d_emb` must be bigger than or equal to `1`.')

        if d_hid < 1:
            raise ValueError('`d_hid` must be bigger than or equal to `1`.')

        if not 0 <= dropout <= 1:
            raise ValueError('`dropout` must range from `0.0` to `1.0`.')

        if num_linear_layers < 1:
            raise ValueError(
                '`num_linear_layers` must be bigger than or equal to `1`.'
            )

        if num_rnn_layers < 1:
            raise ValueError(
                '`num_rnn_layers` must be bigger than or equal to `1`.'
            )

        if pad_token_id < 0:
            raise ValueError(
                '`pad_token_id` must be bigger than or equal to `0`.'
            )

        if vocab_size < 1:
            raise ValueError(
                '`vocab_size` must be bigger than or equal to `1`.'
            )

        if vocab_size <= pad_token_id:
            raise ValueError(
                '`pad_token_id` must be smaller than `vocab_size`.'
            )

        # Token embedding layer.
        # Dimension: (V, E).
        self.emb_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=pad_token_id
        )
        self.emb_dropout = torch.nn.Dropout(dropout)

        # Project from `d_emb` into `d_hid`.
        # Dimension: (E, H).
        self.proj_emb_to_hid = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=d_emb,
                out_features=d_hid
            ),
            torch.nn.Dropout(dropout)
        )

        # Sequential RNN layer(s).
        # Dimension: (H, H).
        if num_rnn_layers == 1:
            self.rnn_layer = torch.nn.RNN(
                input_size=d_hid,
                hidden_size=d_hid,
                batch_first=True
            )
        else:
            self.rnn_layer = torch.nn.RNN(
                input_size=d_hid,
                hidden_size=d_hid,
                num_layers=num_rnn_layers,
                dropout=dropout,
                batch_first=True
            )

        # Sequential linear layer(s).
        # Dimension: (H, H).
        proj_hid_to_emb = []
        for _ in range(num_linear_layers - 1):
            proj_hid_to_emb.append(torch.nn.Dropout(dropout))
            proj_hid_to_emb.append(
                torch.nn.Linear(
                    in_features=d_hid,
                    out_features=d_hid
                )
            )
            proj_hid_to_emb.append(torch.nn.ReLU())

        # Sequential linear layer(s)' last layer.
        # Dimension: (H, E).
        proj_hid_to_emb.append(torch.nn.Dropout(dropout))
        proj_hid_to_emb.append(
            torch.nn.Linear(
                in_features=d_hid,
                out_features=d_emb
            )
        )
        self.proj_hid_to_emb = torch.nn.Sequential(*proj_hid_to_emb)

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

        # 將每個 hidden vectors 轉換維度至 embedding dimension
        # ht 維度: (B, S, E)
        ht = self.proj_hid_to_emb(ht)

        # 與轉置後的 embedding matrix 進行矩陣乘法取得預測文字
        # 重複使用 embedding matrix 的目的為節省參數數量
        # return 維度: (B, S, V)
        return ht.matmul(self.emb_layer.weight.transpose(0, 1))

    def predict(self, batch_sequences: torch.Tensor) -> torch.Tensor:
        r"""Convert model output logits into prediction.

        Args:
            batch_sequences:
                Batch of sequences which have been encoded by
                `lmp.tokenizer.BaseTokenizer` with numeric type `torch.int64`.

        Raises:
            TypeError:
                When `batch_sequences` is not an instance of `Tensor`.

        Returns:
            Predicition using softmax on model output logits with numeric type `torch.float32`.
        """
        # Type check
        if not isinstance(batch_sequences, torch.Tensor):
            raise TypeError(
                '`batch_sequences` must be an instance of `Tensor`.'
            )

        return torch.nn.functional.softmax(self(batch_sequences), dim=-1)
