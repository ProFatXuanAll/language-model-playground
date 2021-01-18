r"""Transformer Decoder Model."""

import math
from typing import ClassVar, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


class LayerNorm(nn.Module):
    r"""Layer Normalization.

    Applies Layer Normalization over a mini-batch of inputs.

    Parameters
    ==========
    d_hid: int
        Hidden dimension for MLP and Transformer.
        Must be bigger than or equal to ``1``.
    eps: float
        Epsilon, a value added to the denominator for numerical stability.

    Attributes
    ==========
    a_2: torch.nn.parameter.Parameter
        Alpha, a learnable factor in normalization.
    b_2: torch.nn.parameter.Parameter
        Beta, a learnable factor in normalization.
    eps: float
        Epsilon, a value added to the denominator for numerical stability.

    References
    ==========
        Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. 2016 . 
        Layer Normalization.
    """

    def __init__(self, d_hid, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_hid))
        self.b_2 = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, x):
        r"""Perform forward pass.

        Forward pass algorithm is structured as follow:

        .. math::

            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] \
            + \epsilon}} * \alpha + \beta \\

        Parameters
        ==========
        x: torch.Tensor
            Input sequence with shape ``(*, E)``.

        Returns
        =======
        torch.Tensor
            Same shape as input.
        """

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    r"""Positional Encoding.

    In order for the model to make use of the order of the sequence,
    we add “positional encodings” to the input embeddings.

    Parameters
    ==========
    d_hid: int
        Hidden dimension for MLP and Transformer.
        Must be bigger than or equal to ``1``.
    dropout: float
        Dropout probability.
        Must satisfy ``0.0 <= dropout <= 1.0``.
    max_len: int
        Max length of the input sequence.
        Must be bigger than or equal to ``1``.

    Attributes
    ==========
    dropout: torch.nn.Dropout
        Positional Encoding dropout.
    pe: torch.nn.FloatTensor
        The value of Positional Encoding.
    """

    def __init__(self, d_hid, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_hid)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2) *
                             -(math.log(10000.0) / d_hid))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        r"""Perform forward pass.

        Forward pass algorithm is structured as follow:

        .. math::

            \begin{align}
            \text{PE}_{(\text{pos},\text{2i})} &=\
             \sin(\frac{\text{pos}}{10000^\frac{\text{2i}}{d_{\text{model}}}}) \\
            \text{PE}_{(\text{pos},\text{2i+1})} &=\
             \cos(\frac{\text{pos}}{10000^\frac{\text{2i}}{d_{\text{model}}}}) \\
            \end{align}

        Parameters
        ==========
        x: torch.Tensor
            Input sequence with shape ``(*, E)``.

        Returns
        =======
        torch.Tensor
            Same shape as input.

        """
        x = x + torch.tensor(self.pe[:, :x.size(1)],
                             requires_grad=False).to(x.device)
        return self.dropout(x)


class TransformerModel(RNNModel):
    r"""Transformer language model.

    Use tranformer's encoder to implement language model.

    Parameters
    ==========
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_hid: int
        Hidden dimension for MLP and self attention RNN.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Left intended for subclass parameters extension.
    n_hid_lyr: int
        Number of self attention RNN layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_lyr: int
        Number of MLP layers ``+1`` after self attention RNN layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_post_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    n_pre_hid_lyr: int
        Number of MLP layers ``+1`` before self attention RNN layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_pre_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    p_emb: float
        Dropout probability for token embeddings.
        Must satisfy ``0.0 <= p_emb <= 1.0``.
    p_hid: float
        Dropout probability for every hidden representations.
        Must satisfy ``0.0 <= p_hid <= 1.0``.
    max_seq_len: int
        Max length of the input sequence.
        Must be bigger than or equal to ``1``.
    tknzr: lmp.tknzr.BaseTknzr
        Tokenizer instance with attributes ``pad_tkid`` and ``vocab_size``.

    Attributes
    ==========
    n_head: int
        Head number of multihead attention.
    emb: torch.nn.Embedding
        Token embedding lookup matrix.
        Use token ids to lookup token embeddings.
    emb_dp: torch.nn.Dropout
        Token embedding dropout.
        Drop embedding features with probability ``p_emb``.
    encoderlayer: torch.nn.TransformerEncoderLayer
        TransformerEncoderLayer is made up of self-attn and feedforward network.
    tranformerencoder: torch.nn.TransformerEncoder
        Stack of N encoderlayers.
    model_name: ClassVar[str]
        Model name is ``Transformer_Torch``.
        Used for command line argument parsing.
    pad_tkid: int
        Padding token id.
        Used to create attention mask on padding tokens.
    post_hid: torch.nn.Sequential
        Rectified MLP which transform temporal features from hidden dimension
        ``d_hid`` to embedding dimension ``d_emb``.
        Drop rectified units with probability ``p_hid``.
    pre_hid: torch.nn.Sequential
        Rectified MLP which transform token embeddings from embedding
        dimension ``d_emb`` to hidden dimension ``d_hid``.
        Drop rectified units with probability ``p_hid``.
    norm: ``lmp._transformer_torch.LayerNorm``
        Layer normalization.
    pe: ``lmp._transformer_torch.PositionalEncoding``
        Positional Encoding.

    References
    ==========
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
        Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
        2017. Attention is all you need.
    """
    model_name: ClassVar[str] = 'Transformer_Torch'

    def __init__(
            self,
            *,
            n_head: int,
            d_emb: int,
            d_hid: int,
            n_hid_lyr: int,
            n_post_hid_lyr: int,
            n_pre_hid_lyr: int,
            p_emb: float,
            p_hid: float,
            max_seq_len: int,
            tknzr: BaseTknzr,
            **kwargs: Optional[Dict],
    ):
        super().__init__(
            d_emb=d_emb,
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            n_post_hid_lyr=n_post_hid_lyr,
            n_pre_hid_lyr=n_pre_hid_lyr,
            p_emb=p_emb,
            p_hid=p_hid,
            tknzr=tknzr,
            **kwargs,
        )
        self.n_head = n_head
        self.pad_tkid = tknzr.pad_tkid
        self.pe = PositionalEncoding(d_hid, p_hid, max_seq_len)
        self.norm = LayerNorm(d_hid)
        self.encoderlayer = nn.TransformerEncoderLayer(d_hid, n_head)
        self.tranformerencoder = nn.TransformerEncoder(
            self.encoderlayer, n_hid_lyr, self.norm)

    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Create self attention masks for ``batch_prev_tkids``.

        Self attention masks are created as follow:

        #. Create auto-regressive self attention masks (mask everything above
           diagnoal).
           This is needed since at each time step current input can only see
           previous inputs and itself.
           (shape: ``(S, S)``)
        #. Create padding self attention masks (mask every padding token id
           positions).
           This is needed since paddings are meaningless.
           (shape: ``(B, S)``)

        Parameters
        ==========
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor:
            Auto-regressive self attention masks with shape ``(S, S)`` and
            ``dtype == torch.bool``.
            Padding self attention masks with shape ``(B, S)`` and
            ``dtype == torch.bool``.
        """
        # Get input batch sequence length.
        seq_len = batch_prev_tkids.size(-1)

        # Create auto-regressive self attention masks.
        # Need to move tensor to model running device.
        # Output shape: `(S, S)`.
        # Output dtype: `torch.bool`.
        reg_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        reg_mask = torch.triu(reg_mask, diagonal=1)
        reg_mask = reg_mask.to(batch_prev_tkids.device)

        # Create padding self attention masks.
        # Output shape: `(B, S)`.
        # Output dtype: `torch.bool`.
        pad_mask = batch_prev_tkids == self.pad_tkid

        return reg_mask, pad_mask

    def forward(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Same steps as ``lmp.model.RNNModel`` but use Transformer encoder.

        Parameters
        ==========
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Next token logits for each token id in batch.
            Logits has shape ``(B, S, V)`` and ``dtype == torch.float32``.
        """
        # Token embedding lookup.
        # Input  shape: `(B, S)`.
        # Output shape: `(B, S, E)`.
        batch = self.emb(batch_prev_tkids)

        # Token embedding dropout.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, E)`.
        batch = self.emb_dp(batch)

        # Transform from embedding dimension to hidden dimension.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, H)`.
        batch = self.pre_hid(batch)

        # positional encoding
        batch = self.pe(batch)

        # Encode temporal features.
        # Input  shape: `(B, S, H)`.
        # Output shape: `(B, S, H)`.

        mask = self.create_mask(batch_prev_tkids)
        batch = batch.transpose(0, 1)
        batch = self.tranformerencoder(
            batch, mask=mask[0], src_key_padding_mask=mask[1])
        batch = batch.transpose(0, 1).contiguous()

        # Transform from hidden dimension to embedding dimension.
        # Input  shape: `(B, S, H)`.
        # Output shape: `(B, S, E)`.
        batch = self.post_hid(batch)

        # Transform from embedding dimension to vocabulary dimension by
        # multiplying transpose of embedding matrix.
        # Reduce model parameters by sharing embedding matrix with output.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, V)`.

        return batch @ self.emb.weight.transpose(0, 1)
