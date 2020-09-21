r"""Transformer Language Model

DESCRIPTION
    A torch implementation of transformer model [1],
    Code is based on The Annotated Transformer [2] from Harvard NLP.

    [1] Vaswani, Ashish, et al. "Attention is all you need."
    Advances in neural information processing systems. 2017.
    https://arxiv.org/pdf/1706.03762.pdf

    [2] The Annotated Transformer
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import math
import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    r"""PositionalEncoding

    Generate a fixed sequence of vector and apply to input

    Copy from https://nlp.seas.harvard.edu/2018/04/03/attention.html

    Fomula:
        PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})
        PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})

    Examples:
        >>> batch_size, seq_len = 32, 128
        >>> x = torch.rand((batch_size, seq_len, 10))
        >>> pe = PositionalEncoding(d_model=10, dropout=.1)
        >>> encoded_x = pe(x)
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        r"""

        Args:
            d_model:
                Dimension of each vector.
            dropout:
                Dropout probability.
            max_len:
                Max length of input sequence. Defaults to 5000.
        """

        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        r"""Forward pass

        Args:
            input_sequence:
                input sequence of shape (batch size, sequence len, d_model)

        Returns:
            torch.Tensor:
                positional encoded input
        """
        input_sequence += self.positional_encoding[:, :input_sequence.size(-2)]
        return self.dropout(input_sequence)


class Attention(torch.nn.Module):
    r"""
    Attention Layer

    Perform a learnable linear transform on all input tensor (`Query`, `Key` ,`Value`).
    After that, create a score by `Query` and `Key`
    which decide how much `Value` should pass through.

    Fomula:
        Attention(Query, Key, Value, Mask, Weight)
        =Score\times Value'
        =Softmax(Query'\times Key'^T)\times Value'
        =Softmax(W_{Q}(Query)\times W_{K}(Key)^T)\times W_{V}(Value)

    """

    def __init__(self, d_model: int, d_k: int):
        r"""
        Args:
            d_model:
                input size
            d_k:
                linear transform output size
        """
        super().__init__()
        self.w_q = torch.nn.Linear(d_model, d_k)
        self.w_k = torch.nn.Linear(d_model, d_k)
        self.w_v = torch.nn.Linear(d_model, d_k)
        self.d_k = d_k

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor):
        r"""
        Args:
            query:
                input query
            key:
                input key
            value:
                input value
            mask:
                Whenever position of mask is false,
                set corresponding score to -1e9 then apply softmax.
                Simplified fomula : Softmax( mask(Q x K) )
        """
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return torch.nn.functional.softmax(scores, dim=-1) @ value


class MultiHeadAttention(torch.nn.Module):
    """
    Multi Head Attention:

    concatenate multiple attention and reduce results by linear transform

    (B, L, E) (B, L, E) (B, L, E) -> (B, L, E / H) * H -> (B, L, E)
    """

    def __init__(self, E, H, dropout):
        super().__init__()
        self.attentions = torch.nn.ModuleList(
            [Attention(E, E // H) for _ in range(H)])
        self.Wo = torch.nn.Linear(E, E)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
        # (B, L, E) -> (B, L, D)
        # (B, L, E) -> (B, L, H * D) * Wo(H * D, D) -> (B, L, D)
        Z = torch.cat([att(Q, K, V, mask) for att in self.attentions], dim=-1)
        return self.dropout(self.Wo(Z))


class FF(torch.nn.Module):
    """
    (B, L, E) -> (B, L, E)
    (B, L, E) -> RELU( (B, L, E) * w1(E, D) ) * w2(D, E) -> (B, L, E)
    """

    def __init__(self, E, D, dropout):
        super().__init__()
        self.W1 = torch.nn.Linear(E, D)
        self.W2 = torch.nn.Linear(D, E)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.dropout(torch.nn.functional.relu(self.W1(x)))))


class AddNorm(torch.nn.Module):
    """
    Add two tensor and perform Layer Normalize
    (B, L, E) (B, L, E) -> (B, L, E)
    """

    def __init__(self, E):
        super().__init__()
        self.norm = torch.nn.LayerNorm(E)

    def forward(self, x, sub):
        return self.norm(x + sub)


class DecoderLayer(torch.nn.Module):
    """
    (B, L, E) -> (B, L, E)
    """

    def __init__(self, E, H, Dff, dropout):
        super().__init__()
        self.att = MultiHeadAttention(E=E, H=H, dropout=dropout)
        self.addnorm1 = AddNorm(E=E)
        self.addnorm2 = AddNorm(E=E)
        self.ff = FF(E=E, D=Dff, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.addnorm1(x, self.att(x, x, x, mask))
        x = self.addnorm2(x, self.ff(x))
        return x


class Decoder(torch.nn.Module):
    """
    stacked decoder layer
    (B, L, E) -> (B, L, E)
    """

    def __init__(self, E, Dff, H, Dlayer, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(E=E, Dff=Dff, H=H, dropout=dropout) for _ in range(Dlayer)])

    def forward(self, x, mask):
        for e in self.layers:
            x = e(x, mask)
        return x


class SubsequentMask(torch.nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        self.pad_id = pad_id

    def forward(self, src):
        L = src.shape[1]
        src_mask = (src != self.pad_id).unsqueeze(-2)
        subseq_mask = (torch.from_numpy(
            np.triu(np.ones((1, L, L), dtype=np.uint8), k=1)) == 0).to(src.device)
        return src_mask & subseq_mask


class TransformerLanguageModel(torch.nn.Module):
    """
    Transformer Language Model
    """

    def __init__(
        self,
        d_emb: int,
        dropout: float,
        num_linear_layers: int,
        num_rnn_layers: int,
        pad_token_id: int,
        vocab_size: int
    ):
        super().__init__()

        self.subseqmask = SubsequentMask(pad_id=pad_token_id)
        self.pad_id = pad_token_id
        self.pad_id = pad_token_id

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=pad_token_id
        )

        self.pe = PositionalEncoding(d_emb, dropout)

        self.decoder = Decoder(
            E=d_emb,
            Dlayer=num_rnn_layers,
            Dff=num_linear_layers,
            H=8,
            dropout=dropout
        )

    def forward(self, src):
        # (B, L) -> (B, L, E) -> (B, L, E) -> (B, L, V)
        mask = self.subseqmask(src=src)
        src = self.pe(self.embedding(src))
        return self.decoder(src, mask) @ self.embedding.weight.transpose(0, 1)

    def predict(self, x):
        return torch.nn.functional.softmax(self(x), dim=-1)
