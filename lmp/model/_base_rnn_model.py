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
import torch.nn.utils.rnn


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
            batch_sequences: torch.LongTensor
    ) -> torch.FloatTensor:
        r"""Perform forward pass.

        Args:
            batch_sequences:
                Batch of sequences which have been encoded by
                `lmp.tokenizer.BaseTokenizer`.

        Returns:
            Logits for each token in sequences.
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
            batch_sequences: torch.LongTensor
    ) -> torch.FloatTensor:
        r"""Convert model output logits into prediction.

        Args:
            batch_sequences:
                Batch of sequences which have been encoded by
                `lmp.tokenizer.BaseTokenizer`.

        Returns:
            Predicition using softmax on model output logits.
        """
        return torch.nn.functional.softmax(self(batch_sequences), dim=-1)

    def generate(
            self,
            begin_of_sequence: torch.LongTensor,
            beam_width: int = 4,
            max_seq_len: int = 64
    ) -> torch.LongTensor:
        r"""Using beam search algorithm to generate texts.

        In the comment below, we use following symbols to denote the size of
        each tensors:
            B: beam_width
            S: sequence length
            V: vocabulary size

        Args:
            begin_of_sequence:
                Begining of sequence which model will auto-complete.
            beam_width:
                Number of candidate sequences to output.
            max_seq_len:
                Maximum of output sequences length.
        Returns:
            `beam_width` different sequence generated by model.
        """
        # Get running device.
        device = begin_of_sequence.device

        # Get begin sequence length.
        seq_len = begin_of_sequence.size(-1)

        # Generated sequence.
        # Start shape (1, S).
        # Final shape (B, S).
        cur_seq = begin_of_sequence.view(1, seq_len)

        # Accumulated negative log-likelihood. Using log can change consecutive
        # probability multiplication into sum of log probability which can
        # avoid computational underflow. Initialized to zero with shape (B).
        accum_prob = torch.zeros(beam_width).to(device)

        # Number of total prediction: `max_seq_len - seq_len`.
        for _ in range(max_seq_len - seq_len):
            # Model prediction has shape (B, S, V).
            pred_y = self.predict(cur_seq)

            # Record all beams prediction.
            # Each beam will predict `beam_width` different results.
            # So we totally have `beam_width * beam_width` different results.
            top_k_in_all_beams = []
            for out_beam in range(cur_seq.size(0)):
                # Get `beam_width` different prediction from beam `out_beam`.
                # `top_k_prob_in_beam` has shape (B) and
                # `top_k_index_in_beam` has shape (B).
                top_k_prob_in_beam, top_k_index_in_beam = \
                    pred_y[out_beam, -1].topk(
                        k=beam_width,
                        dim=-1
                    )

                # Record each beam's negative log-likelihood and concate
                # next token id based on prediction.
                for in_beam in range(beam_width):
                    # Accumulate negative log-likelihood. Since log out
                    # negative value when input is in range 0~1, we negate it
                    # to be postive.
                    prob = accum_prob[out_beam] - \
                        top_k_prob_in_beam[in_beam].log()
                    prob = prob.unsqueeze(0)

                    # Concate next predicted token id.
                    seq = torch.cat([
                        cur_seq[out_beam],
                        top_k_index_in_beam[in_beam].unsqueeze(0)
                    ], dim=-1).unsqueeze(0)

                    # Record result.
                    top_k_in_all_beams.append({
                        'prob': prob,
                        'seq': seq
                    })

            # Compare each recorded result in all beams. First concate tensor
            # then use `topk` to get the `beam_width` highest prediction in all
            # beams.
            _, top_k_index_in_all_beams = torch.cat([
                beam['prob']
                for beam in top_k_in_all_beams
            ]).topk(k=beam_width, dim=0)

            # Update `cur_seq` which is the `beam_width` highest results.
            cur_seq = torch.cat([
                top_k_in_all_beams[index]['seq']
                for index in top_k_index_in_all_beams
            ], dim=0)

            # Update accumlated negative log-likelihood.
            accum_prob = torch.cat([
                top_k_in_all_beams[index]['prob']
                for index in top_k_index_in_all_beams
            ], dim=0)

        return cur_seq
