r"""Transformer Model

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
                Input query.
            key:
                Input key.
            value:
                Input value.
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
    r"""
    Multi Head Attention Layer

    Parallel apply multiple different attention to the same input,
    combine results by a linear transform.

    """

    def __init__(self, d_model: int, heads: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.attentions = torch.nn.ModuleList(
            [Attention(d_model, d_model // heads) for _ in range(heads)])
        self.w_output = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor):
        r"""
        Concatnate result of all attention output and transform back to original shape.

        Args:
            query:
                Input query.
            key:
                Input key.
            value:
                Input value.
            mask:
                Whenever position of mask is false,
                set corresponding score to -1e9 then apply softmax.
                Simplified fomula : Softmax( mask(Q x K) )
        """
        output = torch.cat([att(query, key, value, mask)
                            for att in self.attentions], dim=-1)
        return self.dropout(self.w_output(output))


class FeedForward(torch.nn.Module):
    r"""
    Feed Forward Layer

    Stack two layer of linear transform combine with relu activation function.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            d_ff:
                Linear dimension.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.w_in = torch.nn.Linear(d_model, d_ff)
        self.w_out = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor.
        """
        return self.dropout(self.w_out(self.dropout(torch.nn.functional.relu(self.w_in(input)))))


class AddNorm(torch.nn.Module):
    r"""
    Add two tensor and perform Layer Normalize
    """

    def __init__(self, d_model: int):
        r"""
        Args:
            d_model:
                Input size.
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, input: torch.Tensor, sub: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor.
            sub:
                Input tensor.
        """

        return self.norm(input + sub)


class DecoderLayer(torch.nn.Module):
    r"""
    Decode input by attention and feed forward layer
    """

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            d_ff:
                Feed forward layer dimension.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model, heads=heads, dropout=dropout)
        self.addnorm1 = AddNorm(d_model=d_model)
        self.addnorm2 = AddNorm(d_model=d_model)
        self.feedforward = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor.
            mask:
                Mask for attention layer.
        """
        out = self.addnorm1(input, self.attention(input, input, input, mask))
        return self.addnorm2(out, self.feedforward(out))


class Decoder(torch.nn.Module):
    r"""
    Stack decoder layers
    """

    def __init__(self, d_model: int, heads: int,  d_ff: int, layers: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            d_ff:
                Feed forward layer dimension.
            layers:
                Number of decoder layers.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(
            d_model=d_model, d_ff=d_ff, heads=heads, dropout=dropout) for _ in range(layers)])

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor to be decode.
            mask:
                Mask for attention layer.
        """
        for decoder in self.layers:
            input = decoder(input, mask)
        return input


class SubsequentMask(torch.nn.Module):
    r"""
    Generate subsequent mask for attention layer.
    """

    def __init__(self, pad_id: int):
        r"""
        Args:
            pad_id:
                Id to be mask out.
        """
        super().__init__()
        self.pad_id = pad_id

    def forward(self, input: torch.Tensor):
        r"""
        Args:
            input:
                Input to generate mask.
        """
        in_size = input.size(1)
        src_mask = (input != self.pad_id).unsqueeze(-2)
        subseq_mask = (torch.triu(torch.ones(
            (1, in_size, in_size)), 1) == 0).to(input.device)
        return src_mask & subseq_mask


class TransformerModel(torch.nn.Module):
    r"""
    A torch implementation of transformer model,
    Code is based on The Annotated Transformer from Harvard NLP.

    This implementation only use decoder of transformer,
    it's intentent to train as a language model.
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
        r"""
        Args:
            d_emb:
                Number of embedded dimension, same as d_model inside implement.
            dropout:
                Dropout probability.
            num_linear_layers:
                Feed forward layer dimension.
            num_rnn_layers:
                Number of decoder layers.
            pad_token_id:
                Id to be mask out.
            vocab_size:
                Size of vacabulary, needed by embedded layer.
        """
        super().__init__()

        self.subseqmask = SubsequentMask(pad_id=pad_token_id)
        self.pad_id = pad_token_id
        self.pad_id = pad_token_id

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=pad_token_id
        )

        self.positional_encoding = PositionalEncoding(d_emb, dropout)

        self.decoder = Decoder(
            d_model=d_emb,
            layers=num_rnn_layers,
            d_ff=num_linear_layers,
            heads=8,
            dropout=dropout
        )

    def forward(self, input: torch.Tensor):
        r"""
        Args:
            input:
                Batch input to predict next word.
        """
        mask = self.subseqmask(input)
        input = self.positional_encoding(self.embedding(input))
        return self.decoder(input, mask) @ self.embedding.weight.transpose(0, 1)

    def predict(self, input: torch.Tensor):
        r"""
        Run forward and convert output to probability by apply softmax.

        Args:
            input:
                Batch input to predict next word.
        """
        return torch.nn.functional.softmax(self(input), dim=-1)
