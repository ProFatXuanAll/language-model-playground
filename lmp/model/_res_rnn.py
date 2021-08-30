r"""RNN language model with residual connection."""

from typing import ClassVar, Dict, List, Optional

import torch
import torch.nn as nn

from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


class ResRNNBlock(nn.Module):
    r"""Residual connected RNN blocks.

    Each output of RNN will be added with its input and then dropout with
    probability ``p_hid``.
    Residual connection are used as follow:

    .. math::

        \begin{align*}
        t &\in [1, S] \\
        l &\in [1, L] \\
        h_0^l &= 0 \\
        h_t^l &= \text{RNN}(x_t^l, h_{t-1}^l) \\
        y_t^l &= h_t^l + x_t^l \\
        x_t^{l+1} &= y_t^l
        \end{align*}

    Where :math:`S` means sequence length, :math:`L` means number of layer
    (same as ``n_hid_lyr``), :math:`x_t^l` means input sequence time step
    :math:`t` at layer :math:`l`, :math:`h_t^l` means hidden representation
    time step :math:`t` encoded by recurrent layer :math:`l`, :math:`h_0^l`
    means initial hidden representation of recurrent layer :math:`l`,
    :math:`y_t^l` is the output of time step :math:`t` at layer :math:`l`.

    For comment throughout this class and its subclasses, we use the following
    symbols to denote the shape of tensors:

    - ``B``: Batch size.
    - ``H``: Hidden representation dimension.
    - ``S``: Length of sequence of tokens.

    Parameters
    ==========
    d_hid: int
        Hidden dimension for residual connected RNN.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of residual connected RNN layers.
        Must be bigger than or equal to ``1``.
    p_hid: float
        Dropout probability for every hidden representations.
        Must satisfy ``0.0 <= p_hid <= 1.0``.

    Attributes
    ==========
    dp: torch.nn.ModuleList[torch.nn.Dropout]
        Drop each output temporal features of ``self.recur`` with
        probability ``p_hid``.
        Do not dropout last temporal features output from ``self.recur[-1]``
        since :py:class:`lmp.model.ResRNNModel` have ``self.post_hid`` which
        drop output of ``self.hid``.
    recur: torch.nn.ModuleList[torch.nn.RNN]
        List of vanilla RNN which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
    """

    def __init__(
            self,
            *,
            d_hid: int,
            n_hid_lyr: int,
            p_hid: float,
            **kwargs: Optional[Dict],
    ):
        super().__init__()

        # Create vanilla RNN layers and put in module list.
        # RNN in `self.recur` are treated as sequential RNN.
        # Input tensor : Output of `ResRNNModel.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.recur = nn.ModuleList([
            nn.RNN(input_size=d_hid, hidden_size=d_hid, batch_first=True)
            for _ in range(n_hid_lyr)
        ])

        # Create dropout layers.
        # Only need to create `n_hid_lyr - 1` since `ResRNNModel.post_hid`
        # drop output of `ResRNNModel.hid`.
        # Input tensor : Output of `self.recur`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of sparse recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype: `torch.float32`.
        dp: List[nn.Module] = [
            nn.Dropout(p=p_hid)
            for _ in range(n_hid_lyr - 1)
        ]
        dp.append(nn.Identity())
        self.dp = nn.ModuleList(dp)

    def forward(self, batch_tk_reps: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Forward pass algorithm is structured as follow:

        #. Input batch of previous token hidden representations.
           (shape: ``(B, S, H)``)
        #. Pair block and dropouts in ``self.recur`` and ``self.dp``.
           Pair only first ``n_hid_lyr - 1`` blocks and dropouts.
           Use for-loop to perform the following operations:

           #. Use paired block to encode temporal features.
              (shape: ``(B, S, H)``)
           #. Add input and output of paired block.
              (shape: ``(B, S, H)``)
           #. Use paired dropout to drop some features.
              This step is skipped on last for loop step.
              (shape: ``(B, S, H)``)
           #. Use sparse features as input of next loop.

        #. Return final output.
           (shape: ``(B, S, H)``)

        Parameters
        ==========
        batch_tk_reps: torch.Tensor
            Batch of previous token hidden representation.
            ``batch_tk_reps`` has shape ``(B, S, H)`` and
            ``dtype == torch.float32``.

        Returns
        =======
        torch.Tensor
            Temporal features with shape ``(B, S, H)`` and
            ``dtype == torch.float32``.
        """
        # Initialize input of first block.
        batch_in = batch_tk_reps

        # Pair recurrent layer and dropout.
        for recur, dp in zip(self.recur, self.dp):
            # Encode temporal features.
            # Input shape : `(B, S, H)`.
            # Output shape: `(B, S, H)`.
            batch_out, _ = recur(batch_in)

            # Drop features on output of residual connection.
            # Input shape : `(B, S, H)`.
            # Output shape: `(B, S, H)`.
            batch_out = dp(batch_out + batch_in)

            # Replace input of next block with output of previous block.
            batch_in = batch_out

        return batch_out


class ResRNNModel(RNNModel):
    r"""RNN language model with residual connection.

    Same architecture as :py:class:`lmp.model.RNNModel` but use residual
    connection on RNN layer.

    Parameters
    ==========
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_hid: int
        Hidden dimension for MLP and residual connected RNN.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of residual connected RNN layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_lyr: int
        Number of MLP layers ``+1`` after residual connected RNN layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_post_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    n_pre_hid_lyr: int
        Number of MLP layers ``+1`` before residual connected RNN layer.
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
    emb: torch.nn.Embedding
        Token embedding lookup matrix.
        Use token ids to lookup token embeddings.
    emb_dp: torch.nn.Dropout
        Token embedding dropout.
        Drop embedding features with probability ``p_emb``.
    hid: lmp.model.ResRNNBlock
        Residual connected RNN which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Drop temporal features with probability ``p_hid``.
    model_name: ClassVar[str]
        Model name is ``res-RNN``.
        Used for command line argument parsing.
    post_hid: torch.nn.Sequential
        Rectified MLP which transform temporal features from hidden dimension
        ``d_hid`` to embedding dimension ``d_emb``.
        Drop rectified units with probability ``p_hid``.
    pre_hid: torch.nn.Sequential
        Rectified MLP which transform token embeddings from embedding
        dimension ``d_emb`` to hidden dimension ``d_hid``.
        Drop rectified units with probability ``p_hid``.
    """
    model_name: ClassVar[str] = 'res-RNN'

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

        # Override RNN layer with residual connected RNN.
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.hid = ResRNNBlock(
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            p_hid=p_hid,
            **kwargs,
        )

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
        #. Use ``self.hid`` to encode temporal features.
           (shape: ``(B, S, H)``)
        #. Use ``self.post_hid`` to transform temporal features from hidden
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

        # Encode temporal features.
        # Input  shape: `(B, S, H)`.
        # Output shape: `(B, S, H)`.
        batch = self.hid(batch)

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
