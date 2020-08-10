r"""Language model with RNN layers.

Usage:
    model = lmp.model.BaseRNNModel(...)
    logits = model(...)
    pred = model.predict(...)
    gen_seq = model.generate(...)
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

    Each input token will first be embedded into vectors, then sequentially
    feed into RNN layers. Output vectors of RNN layer then go through
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
        super().__init__()

        # Embedding layer
        # Dimension: (V, E)
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=pad_token_id
        )

        # Sequential RNN layer
        # Dimension: (E, H)
        self.rnn_layer = torch.nn.RNN(
            input_size=d_emb,
            hidden_size=d_hid,
            num_layers=num_rnn_layers,
            dropout=dropout,
            batch_first=True
        )

        # Sequential linear layer
        # Dimension: (H, E)
        linear_layers = []

        for _ in range(num_linear_layers):
            linear_layers.append(
                torch.nn.Linear(
                    in_features=d_hid,
                    out_features=d_hid
                )
            )
            linear_layers.append(
                torch.nn.ReLU()
            )
            linear_layers.append(
                torch.nn.Dropout(dropout)
            )

        linear_layers.append(
            torch.nn.Linear(
                in_features=d_hid,
                out_features=d_emb
            )
        )

        self.linear_layers = torch.nn.Sequential(*linear_layers)

    def forward(
            self,
            batch_sequences: torch.Tensor
    ) -> torch.Tensor:
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
        batch_sequences = self.embedding_layer(batch_sequences)

        # 將每個 embedding vectors 依序經由 RNN 輸入 得到輸出 hidden vectors
        # ht 維度: (B, S, H)
        ht, _ = self.rnn_layer(batch_sequences)

        # 將每個 hidden vectors 轉換維度至 embedding dimension
        # ht 維度: (B, S, E)
        ht = self.linear_layers(ht)

        # 與轉置後的 embedding matrix 進行矩陣乘法取得預測文字
        # 重複使用 embedding matrix 的目的為節省參數數量
        # yt 維度: (B, S, V)
        yt = ht.matmul(self.embedding_layer.weight.transpose(0, 1))

        return yt

    def predict(
            self,
            batch_sequences: torch.Tensor
    ) -> torch.Tensor:
        r"""Convert model output logits into prediction.

        Args:
            batch_sequences:
                Batch of sequences which have been encoded by
                `lmp.tokenizer.BaseTokenizer` with numeric type `torch.int64`.

        Returns:
            Predicition using softmax on model output logits with numeric type `torch.float32`.
        """
        return torch.nn.functional.softmax(self(batch_sequences), dim=-1)
