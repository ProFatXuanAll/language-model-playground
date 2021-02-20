r"""Transformer Language Model with nn.TranformerEncoderlayer."""


import argparse
import math
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


class PositionalEncoding(nn.Module):
    r"""Positional Encoding.

    In order for Transformer model to make use of postional information of
    input sequence, we usually add positional encodings to the input
    embeddings.

    .. math::

        \begin{align}
        \text{PE}_{(\text{pos},\text{2i})} &= \sin\
            (\frac{\text{pos}}{10000^\frac{\text{2i}}{d_{\text{emb}}}}) \\
        \text{PE}_{(\text{pos},\text{2i+1})} &= \cos\
            (\frac{\text{pos}}{10000^\frac{\text{2i}}{d_{\text{emb}}}}) \\
        \end{align}

    Parameters
    ==========
    d_emb: int
        embden dimension for MLP and Transformer.
        Must be bigger than or equal to ``1``.
    dropout: float
        Dropout probability.
        Must satisfy ``0.0 <= dropout <= 1.0``.
    max_seq_len: int
        Max length of the input sequence.
        Must be bigger than or equal to ``1``.

    Attributes
    ==========
    dropout: torch.nn.Dropout
        Dropout after Positional Encoding.
    pe: torch.Tensor
        The values of Positional Encoding.
    """

    def __init__(self, d_emb: int, dropout: float, max_seq_len: int):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding table.
        # Shape : `(S, H)`
        self.pe = torch.zeros(max_seq_len, d_emb)

        # Position order from `0` to `S - 1`.
        # Shape: `(S, 1)`
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        # Compute the positional encodings once in log space.
        # Shape : `(1, S, H)`
        div_term = torch.exp(torch.arange(0, d_emb, 2) *
                             -(math.log(10000.0) / d_emb))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Parameters
        ==========
        src: torch.Tensor
            Input sequence with shape ``(B, S, E)``.

        Returns
        =======
        torch.Tensor
            Same shape as input.

        """
        # Add positional encoding to each sequences.
        # Input shape : `(B, S, H)`
        # Output shape : `(B, S, H)`
        pe = self.pe.detach().to(src.device)
        output = src + pe[:, :src.size(1)]
        return self.dropout(output)


class TransformerModel(BaseModel):
    r"""Transformer language model.

    Use Tranformer's encoder with masks to implement language model.

    Parameters
    ==========
    n_head: int
        Head number of multihead attention.
        ``d_emb`` should be divisible by it.
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_ff: int
        Feed forward layer dimension.
        Must be bigger than or equal to ``1``.
    n_hid_lyr: int
        Number of Tranformer's encoder layers.
        Must be bigger than or equal to ``1``.
    p_emb: float
        Dropout probability for token embeddings.
        Must satisfy ``0.0 <= p_emb <= 1.0``.
    p_hid: float
        Dropout probability for each Transformerlayer.
        Must satisfy ``0.0 <= p_emb <= 1.0``.
    max_seq_len: int
        Max length of the input sequence.
        Must be bigger than or equal to ``1``.
    tknzr: lmp.tknzr.BaseTknzr
        Tokenizer instance with attributes ``pad_tkid`` and ``vocab_size``.
    kwargs: Dict, optional
        Useless parameter.
        Left intended for subclass parameters extension.

    Attributes
    ==========
    emb: torch.nn.Embedding
        Token embedding lookup matrix.
        Use token ids to lookup token embeddings.
    emb_dp: torch.nn.Dropout
        Token embedding dropout.
        Drop embedding features with probability ``p_emb``.
    model_name: ClassVar[str]
        Model name is ``Transformer``.
        Used for command line argument parsing.
    pad_tkid: int
        Padding token id.
        Used to create attention mask on padding tokens.
    pe: lmp.model.PositionalEncoding
        Positional Encoding.
    encoderlayer: torch.nn.TransformerEncoderLayer
        TransformerEncoderLayer is made up of ``MultiHeadAttention`` layer
        and ``Feedforward`` layer.
    tranformerencoder: torch.nn.TransformerEncoder
        Stack of N encoderlayers.

    References
    ==========
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
        Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
        2017. Attention is all you need.
    """
    model_name: ClassVar[str] = 'Transformer'

    def __init__(
            self,
            *,
            n_head: int,
            d_emb: int,
            d_ff: int,
            n_hid_lyr: int,
            p_emb: float,
            p_hid: float,
            max_seq_len: int,
            tknzr: BaseTknzr,
            **kwargs: Optional[Dict],
    ):
        super().__init__()

        if not isinstance(n_head, int):
            raise TypeError('`n_head` must be an instance of `int`')
        if not isinstance(max_seq_len, int):
            raise TypeError('`max_seq_len` must be an instance of `int`')
        if d_emb % n_head != 0:
            raise ValueError('`d_emb` must be divisible by `n_head`.')

        self.pad_tkid = tknzr.pad_tkid

        # Token embedding layer.
        # Use token ids to lookup token embeddings.
        # Input tensor : Batch of token ids.
        # Input shape  : `(B, S)`.
        # Input dtype  : `torch.int64`.
        # Output tensor: Batch of token embeddings.
        # Output shape : `(B, S, E)`.
        # Output dtype : `torch.float32`.
        self.emb = nn.Embedding(
            num_embeddings=tknzr.vocab_size,
            embedding_dim=d_emb,
            padding_idx=self.pad_tkid,
        )

        # Token embedding dropout layer.
        # Drop embedding features with probability `p_emb`.
        # Input tensor : Output of `self.emb`.
        # Input shape  : `(B, S, E)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of sparse token embeddings.
        # Output shape : `(B, S, E)`.
        # Output dtype : `torch.float32`.
        self.emb_dp = nn.Dropout(p=p_emb)

        # Positional Encoding
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, E)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of sequences with Positional Encoding.
        # Output shape : `(B, S, E)`.
        # Output dtype : `torch.float32`.
        self.pe = PositionalEncoding(d_emb, p_emb, max_seq_len)

        # A sigle layer architecture of Transformer encoder,
        # Including a `MutiheadAttention`, `FeedForward`, `LayerNorm`
        #  with dropouts.
        self.encoderlayer = nn.TransformerEncoderLayer(
            d_emb, n_head, dropout=p_hid, dim_feedforward=d_ff)

        # Stack of `n_hid_lyr` encoder layers.
        # Input tensor : Output of `self.pe`.
        # Input shape  : `(S, B, E)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of Transformer encoder output.
        # Output shape : `(S, B, E)`.
        # Output dtype : `torch.float32`.
        self.tranformerencoder = nn.TransformerEncoder(
            self.encoderlayer, n_hid_lyr)

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
        pad_mask = pad_mask.to(batch_prev_tkids.device)

        return reg_mask, pad_mask

    def forward(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Forward pass algorithm is structured as follow:

        #. Input batch of previous token ids.
           (shape: ``(B, S)``)
        #. Use batch of previous token ids to perform token embeddings lookup
           on ``self.emb``.
           (shape: ``(B, S, E)``)
        #. Use ``self.emb_dp`` to drop some features in token embeddings.
           (shape: ``(B, S, E)``)
        #. Use ``self.pe`` to add positional encoding to batch of inputs.
           (shape: ``(B, S, E)``)
        #. Use ``torch.transpose`` to transform to shape model need.
           (shape: ``(S, B, E)``)
        #. Use ``self.tranformerencoder`` to encode features.
           (shape: ``(S, B, E)``)
        #. Use ``torch.transpose`` to transform to shape model need.
           (shape: ``(B, S, E)``)
        #. Find the most possible next token id in embedding matrix
           ``self.emb`` using inner product.
           This reduce parameters since we share weight on token embedding and
           output projection.
           (shape: ``(B, S, V)``)
        #. Return logits.
           Used with ``self.pred`` to convert logit into prediction.
           Used wtih ``self.loss_fn`` to perform optimization.
           (shape: ``(B, S, V)``)

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

        # Add positional encoding to ecah sequences.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, E)`.
        batch = self.pe(batch)

        # Create auto-regressive and padding mask.
        reg_mask, pad_mask = self.create_mask(batch_prev_tkids)

        # Transform to the accepted size for Transformer encoder.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(S, B, E)`.
        batch = batch.transpose(0, 1)

        # Encode features.
        # Input  shape: `(S, B, E)`.
        # Output shape: `(S, B, E)`.
        batch = self.tranformerencoder(
            batch, mask=reg_mask, src_key_padding_mask=pad_mask)

        # Transform to the origin shape.
        # Input  shape: `(S, B, E)`.
        # Output shape: `(B, S, E)`.
        batch = batch.transpose(0, 1).contiguous()

        # Transform from embedding dimension to vocabulary dimension by
        # multiplying transpose of embedding matrix.
        # Reduce model parameters by sharing embedding matrix with output.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, V)`.
        return batch @ self.emb.weight.transpose(0, 1)

    def loss_fn(
            self,
            batch_next_tkids: torch.Tensor,
            batch_prev_tkids: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate language model training loss.

        Use forward pass to get logits and then use cross-entropy to calculate
        next token prediction loss.
        Use teacher forcing to implement this method.

        Parameters
        ==========
        batch_next_tkids: torch.Tensor
            Prediction targets.
            Batch of next token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_next_tkids`` has same shape and ``dtype`` as
            ``batch_prev_tkids``.
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Average next token prediction loss.
            Returned tensor has shape ``(1)`` and ``dtype == torch.float32``.
        """
        # Forward pass.
        # Input  shape: `(B, S)`.
        # Output shape: `(B, S, V)`.
        logits = self(batch_prev_tkids)

        # Reshape logits to calculate loss.
        # Input  shape: `(B, S, V)`.
        # Output shape: `(BxS, V)`.
        logits = logits.reshape(-1, self.emb.num_embeddings)

        # Reshape target to calculate loss.
        # Input  shape: `(B, S)`.
        # Output shape: `(BxS)`.
        batch_next_tkids = batch_next_tkids.reshape(-1)

        # Loss function of next token prediction.
        # All logits are used since we use teacher forcing to optimize.
        # Logits tensor: Batch of next token prediction logits.
        # Logits shape : `(BxS, V)`.
        # Logits dtype : `torch.float32`.
        # Target tensor: Batch of next token prediction target.
        # Target shape : `(BxS)`.
        # Target dtype : `torch.int64`.
        # Output tensor: Average next tokens prediction loss.
        # Output shape : `(1)`.
        # Output dtype : `torch.float32`.
        return F.cross_entropy(logits, batch_next_tkids)

    def pred(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Next token prediction.

        Use forward pass ouput logits to choose the most possible token id
        from vocabulary as next token.

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
            Softmax predicition for next token.
            Return tensor has shape ``(B, S, V)`` and
            ``dtype == torch.float32``.
        """
        # Forward pass.
        # Input  shape: `(B, S)`.
        # Output shape: `(B, S, V)`.
        logits = self(batch_prev_tkids)

        # Convert logits to probabilities using softmax.
        # Input tensor : Batch of next token prediction logits.
        # Input shape  : `(B, S, V)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of next token prediction probabilities.
        # Output shape : `(B, S, V)`.
        # Output dtype : `torch.float32`.
        return F.softmax(logits, dim=-1)

    @staticmethod
    def train_parser(parser: argparse.ArgumentParser) -> None:
        r"""Training language model CLI arguments parser for ``TransformerModel``.

        Parameters
        ==========
        parser: argparse.ArgumentParser
            Parser for CLI arguments.

        See Also
        ========
        lmp.script.train_model
            Language model training script.

        Examples
        ========
        >>> import argparse
        >>> from lmp.model._transformer import TransformerModel
        >>> parser = argparse.ArgumentParser()
        >>> TransformerModel.train_parser(parser)
        >>> args = parser.parse_args([
        ...     '--batch_size', '32',
        ...     '--beta1', '0.9',
        ...     '--beta2', '0.99',
        ...     '--ckpt_step', '1000',
        ...     '--dset_name', 'wikitext-2',
        ...     '--eps', '1e-8',
        ...     '--exp_name', 'my_exp',
        ...     '--log_step', '200',
        ...     '--lr', '1e-4',
        ...     '--max_norm', '1',
        ...     '--max_seq_len', '-1',
        ...     '--n_epoch', '10',
        ...     '--tknzr_exp_name', 'my_tknzr_exp',
        ...     '--ver', 'train',
        ...     '--wd', '1e-2',
        ...     '--n_head', '4',
        ...     '--d_emb', '100',
        ...     '--d_ff', '2048',
        ...     '--n_hid_lyr', '2',
        ...     '--p_emb', '0.1',
        ...     '--p_hid', '0.1',
        ... ])
        >>> args.n_head == 4
        True
        >>> args.d_emb == 100
        True
        >>> args.d_ff == 2048
        True
        >>> args.n_hid_lyr == 2
        True
        >>> args.p_emb == 0.1
        True
        >>> args.p_hid == 0.1
        True
        """
        # Load common arguments.
        BaseModel.train_parser(parser=parser)

        # Required arguments.
        group = parser.add_argument_group('TransformerModel arguments')
        group.add_argument(
            '--n_head',
            help='number of Transformer heads.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--d_emb',
            help='Token embedding dimension.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--d_ff',
            help='Feed forward layer dimension.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--n_hid_lyr',
            help='Number of Tranformer layers.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--p_emb',
            help='Dropout probability for token embeddings.',
            required=True,
            type=float,
        )
        group.add_argument(
            '--p_hid',
            help='Dropout probability for each Transformerlayer.',
            required=True,
            type=float,
        )
