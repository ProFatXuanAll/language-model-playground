r"""Residual connected RNN language model with self attention mechanism."""

import math
from typing import ClassVar, Dict, Optional

import torch
import torch.nn.functional as F

from lmp.model._sattn_rnn import SAttnRNNBlock, SAttnRNNModel
from lmp.tknzr._base import BaseTknzr


class ResSAttnRNNBlock(SAttnRNNBlock):
    r"""Residual connected RNN block with self attention mechanism.

    Same architecture as :py:class:`lmp.model.SAttnRNNModel` but use residual
    connection.
    Residual connection is used as follow:

    .. math::

        \begin{align*}
        t &\in [1, S] \\
        l &\in [1, L] \\
        h_0^l &= 0 \\
        y_t^l &= \text{SAttnRNN}(x_t^l, h_{t-1}^l) + x_t^l \\
        x_{t+1}^l &= y_t^l
        \end{align*}

    Where :math:`S` means sequence length, :math:`L` means number of layer
    (same as ``n_hid_lyr``), :math:`x_t^l` means input sequence time step
    :math:`t` at layer :math:`l`, :math:`h_t^l` means hidden representation
    time step :math:`t` encoded by self attention recurrent layer :math:`l`,
    :math:`h_0^l` means initial hidden representation of self attention
    recurrent layer :math:`l`, :math:`y_t^l` is the output of time step
    :math:`t` at layer :math:`l`.

    Each output of RNN will then use self attention scores as weights to
    calculate weighted sum as final output.
    Residual connection is added to final output.

    Parameters
    ==========
    d_hid: int
        Hidden dimension for RNN and self attention linear transformation
        weights (including query, key, value and output).
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of residual connected self attention RNN layers.
        Must be bigger than or equal to ``1``.
    p_hid: float
        Dropout probability for every hidden representations.
        Must satisfy ``0.0 <= p_hid <= 1.0``.
    """

    def forward(
            self,
            batch_tk_mask: torch.Tensor,
            batch_tk_reps: torch.Tensor,
    ) -> torch.Tensor:
        r"""Perform forward pass.

        Forward pass algorithm is structured as follow:

        #. Input batch of previous token hidden representations.
           (shape: ``(B, S, H)``)
        #. Use for-loop to perform the following operations:

           #. Use recurrent layer to encode temporal features.
              (shape: ``(B, S, H)``)
           #. Calculate query, key and value features on recurrent layer
              output.
              (shape: ``(B, S, H)``)
           #. Calculate self attention scores with query and key features.
              (shape: ``(B, S, S)``)
           #. Mask parts of self attention scores by replacing masked positions
              with large negative value.
              (shape: ``(B, S, S)``)
           #. Use self attention scores to as weights to calculate weighted sum
              on value features.
              (shape: ``(B, S, H)``)
           #. Perform one more linear transformation on weighted sum features.
              (shape: ``(B, S, H)``)
           #. Add residual connection to previous step.
              (shape: ``(B, S, H)``)
           #. Drop some features.
              This step is skipped on last for loop step.
              (shape: ``(B, S, H)``)
           #. Use sparse features as input of next loop.

        #. Return final output.
           (shape: ``(B, S, H)``)

        Parameters
        ==========
        batch_tk_mask: torch.Tensor
            Batch of attention mask.
            ``batch_tk_mask`` has shape ``(B, S, S)`` and
            ``dtype == torch.bool``.
        batch_tk_reps: torch.Tensor
            Batch of previous token hidden representation.
            ``batch_tk_reps`` has shape ``(B, S, H)`` and
            ``dtype == torch.float32``.

        Returns
        =======
        torch.Tensor
            Residual connected self attention recurrent features with shape
            ``(B, S, H)`` and ``dtype == torch.float32``.
        """
        # Initialize input of first block.
        batch = batch_tk_reps

        # Self attention RNN loops.
        for (recur, query, key, value, out, dp) in zip(
            self.recur,
            self.query,
            self.key,
            self.value,
            self.out,
            self.dp,
        ):
            # Encode temporal features.
            # Input  shape: `(B, S, H)`.
            # Output shape: `(B, S, H)`.
            batch_recur, _ = recur(batch)

            # Transform temporal features to query, key and value features.
            # Input  shape: `(B, S, H)`.
            # Output shape: `(B, S, H)`.
            q = query(batch_recur)
            k = key(batch_recur)
            v = value(batch_recur)

            # Calculate self attention scores with query and key features.
            # Self attention scores are scaled down by hidden dimension square
            # root to avoid overflow.
            # Input  shape: `(B, S, H)`.
            # Output shape: `(B, S, S)`.
            attn = q @ k.transpose(-1, -2) / math.sqrt(batch.size(-1))

            # Mask parts of attention scores by replacing with large negative
            # values.
            # Input  shape: `(B, S, S)`.
            # Output shape: `(B, S, S)`.
            attn.masked_fill_(batch_tk_mask, -1e9)

            # Softmax normalize on attention scores.
            # Large negative values will be closed to zero after normalization.
            # Input  shape: `(B, S, S)`.
            # Output shape: `(B, S, S)`.
            attn = F.softmax(attn, dim=-1)

            # Use attention scores to calculate weighted sum on value features.
            # Then perform one more linear tranformation on weighted sum.
            # Finally add residual connection and dropout some features.
            # Input  shape: `(B, S, S)`.
            # Output shape: `(B, S, H)`.
            batch = dp(out(attn @ v) + batch)

        return batch


class ResSAttnRNNModel(SAttnRNNModel):
    r"""Residual connected RNN language model with self attention mechanism.

    Same architecture as :py:class:`lmp.model.SAttnRNNModel` but use residual
    connection on self attention RNN layer.

    Parameters
    ==========
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_hid: int
        Hidden dimension for MLP and residual connected self attention RNN.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of residual connected self attention RNN layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_lyr: int
        Number of MLP layers ``+1`` after residual connected self attention RNN
        layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_post_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    n_pre_hid_lyr: int
        Number of MLP layers ``+1`` before residual connected self attention
        RNN layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_pre_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    p_emb: float
        Dropout probability for token embeddings.
        Must satisfy ``0.0 <= p_emb <= 1.0``.
    p_hid: float
        Dropout probability for every hidden representations.
        Must satisfy ``0.0 <= p_hid <= 1.0``.
    tknzr: lmp.tknzr.BaseTknzr
        Tokenizer instance with attributes ``pad_tkid`` and ``vocab_size``.

    Attributes
    ==========
    hid: lmp.model.SAttnRNNBlock
        Self attention RNN with residual connection which encode temporal
        features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Drop temporal features with probability ``p_hid``.
    model_name: ClassVar[str]
        Model name is ``res-sattn-RNN``.
        Used for command line argument parsing.
    """
    model_name: ClassVar[str] = 'res-sattn-RNN'

    def __init__(
            self,
            *,
            d_emb: int,
            d_hid: int,
            n_hid_lyr: int,
            n_post_hid_lyr: int,
            n_pre_hid_lyr: int,
            p_emb: float,
            p_hid: float,
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

        # Override self attention RNN layer with residual connected self
        # attention RNN.
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.hid = ResSAttnRNNBlock(
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            p_hid=p_hid,
            **kwargs,
        )
