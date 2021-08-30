r"""RNN language model with self attention mechanism."""

import math
from typing import ClassVar, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


class SAttnRNNBlock(nn.Module):
    r"""RNN block with self attention mechanism.

    Each output of RNN will be used to calculate self attention scores.
    Self attention scores are calculated as follow (see [Vaswani2017]_ for more
    details on self attention scores):

    .. math::

        \begin{align*}
        t &\in [1, S] \\
        l &\in [1, L] \\
        x_{1:S}^l &= x_1^l, \dots, x_S^l \\
        h_0^l &= 0 \\
        h_t^l &= \text{RNN}(x_t^l, h_{t-1}^l) \\
        Q^l &= W_Q^l h_{1:S}^l + b_Q^l \\
        K^l &= W_K^l h_{1:S}^l + b_K^l \\
        V^l &= W_V^l h_{1:S}^l + b_V^l \\
        A^l &= \text{softmax}(\frac{Q^l (K^{l})^{\top}}{\sqrt{S}})V^l \\
        y_{1:S}^l &= W_O^l A^l + b_O^l \\
        x_{1:S}^{l+1} &= y_{1:S}^l
        \end{align*}

    Where :math:`x_{1:S}^l` is the input sequeence with length :math:`S` at
    layer :math:`l`, :math:`L` means number of layer (same as ``n_hid_lyr``).
    :math:`W_Q^l, W_K^l, W_V^l, W_O^l` and :math:`b_Q^l, b_K^l, b_V^l, b_O^l`
    are linear transformation weights and biases for query, key, value, and
    output at layer :math:`l`, respectively.
    :math:`A^l` is self attention scores weighted sum.
    :math:`x_t^l` means input sequence time step :math:`t` at layer :math:`l`,
    :math:`h_t^l` means hidden representation encoded by recurrent layer
    :math:`l`, :math:`h_0^l` means initial hidden representation of recurrent
    layer :math:`l`, :math:`y_{1:t}^l` is the output of layer :math:`l`.

    Each output of RNN will then use self attention scores as weights to
    calculate weighted sum as final output.
    No residual connection is used.

    For comment throughout this class and its subclasses, we use the following
    symbols to denote the shape of tensors:

    - ``B``: Batch size.
    - ``H``: Hidden representation dimension.
    - ``S``: Length of sequence of tokens.

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
        Number of self attention RNN layers.
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
        since :py:class:`lmp.model.SAttnRNNModel` have ``self.post_hid`` which
        drop output of ``self.hid``.
    key: torch.nn.ModuleList[torch.nn.Linear]
        Linear transformation which transform temporal features to key (query
        target) features.
        See [Vaswani2017]_ for details on self attention key.
    out: torch.nn.ModuleList[torch.nn.Linear]
        Final linear transformation after weighted sum using attention scores.
        Do not dropout last output features from ``self.out[-1]`` since
        :py:class:`lmp.model.SAttnRNNModel` have ``self.post_hid`` which
        drop output of ``self.hid``.
        See [Vaswani2017]_ for details on multi-heads self attention.
    query: torch.nn.ModuleList[torch.nn.Linear]
        Linear transformation which transform temporal features to query
        features.
        See [Vaswani2017]_ for details on self attention query.
    recur: torch.nn.ModuleList[torch.nn.RNN]
        List of vanilla RNN which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
    value: torch.nn.ModuleList[torch.nn.Linear]
        Linear transformation which transform temporal features to attention
        scores weighted sum features.
        See [Vaswani2017]_ for details on self attention value.

    References
    ==========
    .. [Vaswani2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
        Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia
        Polosukhin. 2017. Attention is all you need.
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
        # Input tensor : Output of `SAttnRNNModel.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.recur = nn.ModuleList([
            nn.RNN(input_size=d_hid, hidden_size=d_hid, batch_first=True)
            for _ in range(n_hid_lyr)
        ])

        # Create self attention query, key and value transformation layers and
        # put in module list.
        # Input tensor : Output of `self.recur`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Query, key and value token features.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.query = nn.ModuleList([
            nn.Linear(in_features=d_hid, out_features=d_hid)
            for _ in range(n_hid_lyr)
        ])
        self.key = nn.ModuleList([
            nn.Linear(in_features=d_hid, out_features=d_hid)
            for _ in range(n_hid_lyr)
        ])
        self.value = nn.ModuleList([
            nn.Linear(in_features=d_hid, out_features=d_hid)
            for _ in range(n_hid_lyr)
        ])

        # Create self attention final output transformation layers and put in
        # module list.
        # Input tensor : Output of self attention weighted sum.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Linear transformation on self attention weighted sum.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.out = nn.ModuleList([
            nn.Linear(in_features=d_hid, out_features=d_hid)
            for _ in range(n_hid_lyr)
        ])

        # Create dropout layers.
        # Only need to create `n_hid_lyr - 1` since `SAttnRNNModel.post_hid`
        # drop output of `SAttnRNNModel.hid`.
        # Input tensor : Output of `self.out`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Sparse output of `self.out`.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        dp: List[nn.Module] = [
            nn.Dropout(p=p_hid)
            for _ in range(n_hid_lyr - 1)
        ]
        dp.append(nn.Identity())
        self.dp = nn.ModuleList(dp)

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
            Self attention recurrent features with shape ``(B, S, H)`` and
            ``dtype == torch.float32``.
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
            # Finally dropout transformed features.
            # Input  shape: `(B, S, S)`.
            # Output shape: `(B, S, H)`.
            batch = dp(out(attn @ v))

        return batch


class SAttnRNNModel(RNNModel):
    r"""RNN language model with self attention mechanism.

    Same architecture as :py:class:`lmp.model.RNNModel` but use self attention
    on RNN layer.

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
        Intently left for subclass parameters extension.
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
    hid: lmp.model.SAttnRNNBlock
        Self attention RNN which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Drop temporal features with probability ``p_hid``.
    model_name: ClassVar[str]
        Model name is ``sattn-RNN``.
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
    """
    model_name: ClassVar[str] = 'sattn-RNN'

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
        self.pad_tkid = tknzr.pad_tkid

        # Override RNN layer with self attention RNN.
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.hid = SAttnRNNBlock(
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            p_hid=p_hid,
            **kwargs,
        )

    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Create self attention masks for ``batch_prev_tkids``.

        Self attention masks are created as follow:

        #. Create auto-regressive self attention masks (mask everything above
           diagnoal).
           This is needed since at each time step current input can only see
           previous inputs and itself.
           (shape: ``(1, S, S)``)
        #. Create padding self attention masks (mask every padding token id
           positions).
           This is needed since paddings are meaningless.
           (shape: ``(B, S, 1)``)
        #. Perform ``or`` operation on auto-regressive self attention masks and
           padding self attention masks.
           Return ``or`` operation results.
           (shape: ``(B, S, S)``)

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
            Self attention masks with shape ``(B, S, S)`` and
            ``dtype == torch.bool``.
        """
        # Get input batch sequence length.
        seq_len = batch_prev_tkids.size(-1)

        # Create auto-regressive self attention masks.
        # Need to move tensor to model running device.
        # Output shape: `(1, S, S)`.
        # Output dtype: `torch.bool`.
        reg_mask = torch.ones((1, seq_len, seq_len), dtype=torch.bool)
        reg_mask = torch.triu(reg_mask, diagonal=1)
        reg_mask = reg_mask.to(batch_prev_tkids.device)

        # Create padding self attention masks.
        # Output shape: `(B, S, 1)`.
        # Output dtype: `torch.bool`.
        pad_mask = batch_prev_tkids == self.pad_tkid
        pad_mask = pad_mask.unsqueeze(-1)

        # Combine two self attention masks with `or`.
        # Output shape: `(B, S, S)`.
        # Output dtype: `torch.bool`.
        return reg_mask | pad_mask

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
        batch = self.hid(
            batch_tk_reps=batch,
            batch_tk_mask=self.create_mask(batch_prev_tkids=batch_prev_tkids),
        )

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
