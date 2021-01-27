r"""Transformer Language Model with Transformer's encoder architecture."""


import math
import argparse
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn

from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


class PositionalEncoding(nn.Module):
    r"""Positional Encoding.

    In order for the model to make use of the order of the sequence,
    add ``positional encodings`` to the input embeddings.

    .. math::

            \begin{align}
            \text{PE}_{(\text{pos},\text{2i})} &= \sin\
             (\frac{\text{pos}}{10000^\frac{\text{2i}}{d_{\text{model}}}}) \\
            \text{PE}_{(\text{pos},\text{2i+1})} &= \cos\
             (\frac{\text{pos}}{10000^\frac{\text{2i}}{d_{\text{model}}}}) \\
            \end{align}

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
        The values of Positional Encoding.
    """

    def __init__(self, d_hid, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding table.
        # Shape : `(S, H)`
        self.pe = torch.zeros(max_len, d_hid)

        # Position order from `0` to `S - 1`.
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute the positional encodings once in log space.
        # Shape : `(1, S, H)`
        div_term = torch.exp(torch.arange(0, d_hid, 2) *
                             -(math.log(10000.0) / d_hid))
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
    n_head: int
        Head number of multihead attention.
        ``d_hid`` should be divisible by it.

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
    post_hid: torch.nn.Sequential
        Rectified MLP which transform temporal features from hidden dimension
        ``d_hid`` to embedding dimension ``d_emb``.
        Drop rectified units with probability ``p_hid``.
    pre_hid: torch.nn.Sequential
        Rectified MLP which transform token embeddings from embedding
        dimension ``d_emb`` to hidden dimension ``d_hid``.
        Drop rectified units with probability ``p_hid``.
    pe: lmp.model.PositionalEncoding
        Positional Encoding.
    encoderlayer: torch.nn.TransformerEncoderLayer
        TransformerEncoderLayer is made up of MultiHeadAttention layer
        and Feedforward layer.
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

        if d_hid % n_head != 0:
            raise ValueError('`d_hid` must be divisible by `n_head`.')

        self.pad_tkid = tknzr.pad_tkid

        # Positional Encoding
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of sequences with Positional Encoding.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.pe = PositionalEncoding(d_hid, p_hid, max_seq_len)

        # A sigle layer architecture of Transformer encoder.
        self.encoderlayer = nn.TransformerEncoderLayer(d_hid, n_head)

        # Stack of `n_hid_lyr` encoder layers.
        # Input tensor : Output of `self.pe`.
        # Input shape  : `(S, B, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of Transformer encoder output.
        # Output shape : `(S, B, H)`.
        # Output dtype : `torch.float32`.
        self.tranformerencoder = nn.TransformerEncoder(
            self.encoderlayer, n_hid_lyr)

    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Create self attention masks for ``batch_prev_tkids``.

        Self attention masks are created as follow:

        # . Create auto-regressive self attention masks (mask everything above
           diagnoal).
           This is needed since at each time step current input can only see
           previous inputs and itself.
           (shape: ``(S, S)``)
        # . Create padding self attention masks (mask every padding token id
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
        #. Use ``self.pre_hid`` to transform token embeddings from embedding
           dimension ``E`` to hidden dimension ``H``.
           (shape: ``(B, S, H)``)
        #. Use ``self.pe`` to add positional encoding to batch of inputs.
           (shape: ``(B, S, H)``)
        #. Use ``torch.transpose`` to transform to shape model need.
           (shape: ``(S, B, H)``)
        #. Use ``self.tranformerencoder`` to encode features.
           (shape: ``(S, B, H)``)
        #. Use ``torch.transpose`` to transform to shape model need.
           (shape: ``(B, S, H)``)
        #. Use ``self.post_hid`` to transform features from hidden
           dimension ``H`` to embedding dimension ``E``.
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

        # Transform from embedding dimension to hidden dimension.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, H)`.
        batch = self.pre_hid(batch)

        # Add positional encoding to ecah sequences.
        # Input  shape: `(B, S, H)`.
        # Output shape: `(B, S, H)`.
        batch = self.pe(batch)

        # Create auto-regressive and padding mask.
        reg_mask, pad_mask = self.create_mask(batch_prev_tkids)

        # Transform to the accepted size for Transformer encoder.
        # Input  shape: `(B, S, H)`.
        # Output shape: `(S, B, H)`.
        batch = batch.transpose(0, 1)

        # Encode features.
        # Input  shape: `(S, B, H)`.
        # Output shape: `(S, B, H)`.
        batch = self.tranformerencoder(
            batch, mask=reg_mask, src_key_padding_mask=pad_mask)

        # Transform to the origin shape.
        # Input  shape: `(S, B, H)`.
        # Output shape: `(B, S, H)`.
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
        ... ])
        >>> args.batch_size == 32
        True
        >>> args.beta1 == 0.9
        True
        >>> args.beta2 == 0.99
        True
        >>> args.ckpt_step == 1000
        True
        >>> args.dset_name == 'wikitext-2'
        True
        >>> args.eps == 1e-8
        True
        >>> args.exp_name == 'my_exp'
        True
        >>> args.log_step == 200
        True
        >>> args.lr == 1e-4
        True
        >>> args.max_norm == 1
        True
        >>> args.max_seq_len == -1
        True
        >>> args.n_epoch == 10
        True
        >>> args.seed == 42
        True
        >>> args.tknzr_exp_name == 'my_tknzr_exp'
        True
        >>> args.ver == 'train'
        True
        >>> args.wd == 1e-2
        True
        >>> args.n_head == 4
        True
        """
        # Load common arguments.
        RNNModel.train_parser(parser=parser)

        # Required arguments.
        group = parser.add_argument_group('TransformerModel arguments')
        group.add_argument(
            '--n_head',
            help='number of Transformer heads.',
            type=int,
        )
