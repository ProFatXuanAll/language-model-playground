r"""Neural network language model based on vanilla RNN."""

from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn

from lmp.model._base import BaseModel


class RNNModel(BaseModel):
    r"""Neural network language model based on vanilla RNN.

    For comment throughout this class, we use the following symbols to denote
    the shape of tensors:

    - ``B``: Batch size.
    - ``E``: Token embedding dimension.
    - ``H``: Hidden representation dimension.
    - ``S``: Length of sequence of tokens.
    - ``V``: Vocabulary size.

    Use ``self.cal_loss`` for training and use ``self.pred`` for inference.
    Both are depends on forward pass alogorithm ``self.forward``.

    Parameters
    ==========
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_hid: int
        Hidden dimension for MLP and RNN.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intended left for subclass parameters extension.
    n_hid_layer: int
        Number of RNN layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_layer: int
        Number of MLP layers ``+ 1`` after RNN layer.
        Must be bigger than or equal to ``1``.
    n_pre_hid_layer: int
        Number of MLP layers ``+ 1`` before RNN layer.
        Must be bigger than or equal to ``1``.
    n_vocab: int
        Token vocabulary size.
        Must be bigger than or equal to ``1``.
    p_emb: float
        Dropout probability for token embeddings.
        Must satisfy ``0.0 <= p_emb <= 1.0``.
    p_hid: float
        Dropout probability for hidden representation.
        Must satisfy ``0.0 <= p_hid <= 1.0``.
    pad_tkid
        Padding token id.
        Must satisfy ``0 <= pad_tkid <= n_vocab - 1``.

    Attributes
    ==========
    emb: torch.nn.Embedding
        Token embedding lookup matrix.
        Use token ids to lookup token embeddings.
    emb_dp: torch.nn.Dropout
        Token embedding dropout.
        Drop embedding features with probability ``p_emb``.
    hid: torch.nn.RNN
        Vanilla RNN which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Dropout recurrent units if ``n_hid_layer > 1``.
    loss_fn: torch.nn.CrossEntropyLoss
        Loss function of next token prediction.
    model_name: ClassVar[str]
        Display name for model on CLI.
        Only used for command line argument parsing.
    out: torch.nn.Softmax
        Softmax activation which transform logits to prediction.
    post_hid: torch.nn.Sequential
        Rectified MLP which transform temporal features from hidden dimension
        ``d_hid`` to embedding dimension ``d_emb``.
        Drop rectified units with probability ``p_hid``.
    pre_hid: torch.nn.Sequential
        Rectified MLP which transform token embeddings from embedding
        dimension ``d_emb`` to hidden dimension ``d_hid``.
        Drop rectified units with probability ``p_hid``.
    """
    model_name: ClassVar[str] = 'RNN'

    def __init__(
            self,
            *,
            d_emb: int,
            d_hid: int,
            n_hid_layer: int,
            n_post_hid_layer: int,
            n_pre_hid_layer: int,
            n_vocab: int,
            p_emb: float,
            p_hid: float,
            pad_tkid: int,
            **kwargs: Optional[Dict],
    ):
        super().__init__()

        # Token embedding layer.
        # Use token ids to lookup token embeddings.
        # Input              : Batch of token ids.
        # Input shape        : `(B, S)`.
        # Input tensor dtype : `torch.int64`.
        # Output             : Batch of token embeddings.
        # Output shape       : `(B, S, E)`.
        # Output tensor dtype: `torch.float32`.
        self.emb = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=d_emb,
            padding_idx=pad_tkid,
        )

        # Token embedding dropout layer.
        # Drop embedding features with probability `p_emb`.
        # Input              : Output of `self.emb`.
        # Input shape        : `(B, S, E)`.
        # Input tensor dtype : `torch.float32`.
        # Output             : Batch of sparse token embeddings.
        # Output shape       : `(B, S, E)`.
        # Output tensor dtype: `torch.float32`.
        self.emb_dp = nn.Dropout(p=p_emb)

        # Rectified MLP which transform token embeddings from embedding
        # dimension `d_emb` to hidden dimension `d_hid`.
        # Drop rectified units with probability `p_hid`.
        # Input              : Output of `self.emb_dp`.
        # Input shape        : `(B, S, E)`.
        # Input tensor dtype : `torch.float32`.
        # Output             : Batch of sparse rectified token representation.
        # Output shape       : `(B, S, H)`.
        # Output tensor dtype: `torch.float32`.
        pre_hid = [
            nn.Linear(in_features=d_emb, out_features=d_hid),
            nn.ReLU(),
            nn.Dropout(p=p_hid),
        ]

        for _ in range(n_pre_hid_layer):
            pre_hid.append(nn.Linear(in_features=d_hid, out_features=d_hid))
            pre_hid.append(nn.ReLU())
            pre_hid.append(nn.Dropout(p=p_hid))

        self.pre_hid = nn.Sequential(*pre_hid)

        # Vanilla RNN which encode temporal features.
        # Each time step's hidden state depends on current input and previous
        # hidden state.
        # Dropout recurrent units if `n_hid_layer > 1`.
        # Input              : Output of `self.pre_hid`.
        # Input shape        : `(B, S, H)`.
        # Input tensor dtype : `torch.float32`.
        # Output             : Batch of recurrent token hidden states.
        # Output shape       : `(B, S, H)`.
        # Output tensor dtype: `torch.float32`.
        if n_hid_layer == 1:
            self.hid = nn.RNN(
                input_size=d_hid,
                hidden_size=d_hid,
                batch_first=True,
            )
        else:
            self.hid = nn.RNN(
                input_size=d_hid,
                hidden_size=d_hid,
                num_layers=n_hid_layer,
                dropout=p_hid,
                batch_first=True,
            )

        # Rectified MLP which transform temporal features from hidden dimension
        # `d_hid` to embedding dimension `d_emb`.
        # Drop rectified units with probability `p_hid`.
        # Input              : Output of `self.hid`.
        # Input shape        : `(B, S, H)`.
        # Input tensor dtype : `torch.float32`.
        # Output             : Batch of sparse rectified next token embeddings.
        # Output shape       : `(B, S, E)`.
        # Output tensor dtype: `torch.float32`.
        post_hid = []
        for _ in range(n_post_hid_layer):
            post_hid.append(nn.Dropout(p=p_hid))
            post_hid.append(nn.Linear(in_features=d_hid, out_features=d_hid))
            post_hid.append(nn.ReLU())

        post_hid.append(nn.Dropout(p=p_hid))
        post_hid.append(nn.Linear(in_features=d_hid, out_features=d_emb))
        self.post_hid = nn.Sequential(*post_hid)

        # Softmax activation which transform logits to prediction.
        # Input              : Batch of next token prediction logits.
        # Input shape        : `(B, S, V)`.
        # Input tensor dtype : `torch.float32`.
        # Output             : Batch of next token prediction probabilities.
        # Output shape       : `(B, S, V)`.
        # Output tensor dtype: `torch.float32`.
        self.out = nn.Softmax(dim=-1)

        # Loss function of next token prediction.
        # Prediction             : Batch of next token prediction logits.
        # Prediction shape       : `(BxS, V)`.
        # Prediction tensor dtype: `torch.float32`.
        # Target                 : Batch of next token prediction target.
        # Target shape           : `(BxS)`.
        # Target tensor dtype    : `torch.int64`.
        # Output                 : Average next tokens prediction loss.
        # Output shape           : `(1)`.
        # Output tensor dtype    : `torch.float32`.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_tkid: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Forward pass algorithm is structured as follow:

        #. Input batch of token ids.
           (shape: ``(B, S)``)
        #. Use batch of token ids to perform token embeddings lookup on
           ``self.emb``.
           (shape: ``(B, S, E)``)
        #. Use ``self.emb_dp`` to drop some features in token embeddings
           (shape: ``(B, S, E)``)
        #. Use ``self.pre_hid`` to transform token embeddings from embedding
           dimension ``E`` to hidden dimension ``H``.
           (shape: ``(B, S, H)``)
        #. Use ``self.hid`` to encode temporal features.
           (shape: ``(B, S, H)``)
        #. Use ``self.post_hid`` to transform token's recurrent hidden
           representation from hidden dimension ``H`` to embedding dimension
           ``E``.
           (shape: ``(B, S, E)``)
        #. Calculate inner product with weight transpose of ``self.emb``.
           This reduce parameters since we share weight on token embedding and
           output projection.
           (shape: ``(B, S, V)``)
        #. Return logits.
           Use ``self.pred`` to convert logit into prediction.
           Use ``self.cal_loss`` to perform optimization.
           (shape: ``(B, S, V)``)

        Parameters
        ==========
        batch_tkid: torch.Tensor
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_tkid`` has shape ``(B, S)`` and ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Logits for each token in sequences with numeric type `torch.float32`.
        """
        # Token embedding lookup.
        # Input  shape: `(B, S)`.
        # Output shape: `(B, S, E)`.
        batch = self.emb(batch_tkid)

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
        batch, _ = self.hid(batch)

        # Transform from hidden dimension to embedding dimension.
        # Input  shape: `(B, S, H)`.
        # Output shape: `(B, S, E)`.
        batch = self.post_hid(batch)

        # Transform from embedding dimension to vocabulary dimension by
        # multiplying transpose of embedding matrix.
        # We reduce model parameters by sharing embedding matrix with output.
        # Input  shape: `(B, S, E)`.
        # Output shape: `(B, S, V)`.
        return batch @ self.emb.weight.transpose(0, 1)

    def cal_loss(
            self,
            batch_tkid: torch.Tensor,
            batch_next_tkid: torch.Tensor
    ) -> torch.Tensor:
        r"""Calculate language model training loss.

        Use cross-entropy to calculate next token prediction loss.
        Prediction means choose a token from vocabulary as next token.

        Parameters
        ==========
        batch_tkid: torch.Tensor
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_tkid`` has shape ``(B, S)`` and ``dtype == torch.int64``.
        batch_next_tkid: torch.Tensor
            Prediction targets.
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_next_tkid`` has same shape and ``dtype`` as ``batch_tkid``.

        Returns
        =======
        torch.Tensor
            Average next token prediction loss.
        """
        # Forward pass.
        # Input  shape: `(B, S)`.
        # Output shape: `(B, S, V)`.
        logits = self(batch_tkid)

        # Reshape logits to calculate loss.
        # Input  shape: `(B, S, V)`.
        # Output shape: `(BxS, V)`.
        logits = logits.reshape(-1, self.emb.weight.num_embeddings)

        # Reshape target to calculate loss.
        # Input  shape: `(B, S)`.
        # Output shape: `(BxS)`.
        batch_next_tkid = batch_next_tkid.reshape(-1)

        # Calculate average prediction loss.
        # Input  shape: `(BxS, V), (BxS)`.
        # Output shape: `(1)`.
        return self.loss_fn(logits, batch_next_tkid)

    def pred(self, batch_tkid: torch.Tensor) -> torch.Tensor:
        r"""Next token prediction.

        Prediction means choose a token from vocabulary as next token.

        Parameters
        ==========
        batch_tkid: torch.Tensor
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_tkid`` has shape ``(B, S)`` and ``dtype == torch.int64``.

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
        logits = self(batch_tkid)

        # Convert logits to probabilities using softmax.
        # Input  shape: `(B, S, V)`.
        # Output shape: `(B, S, V)`.
        return self.out(logits)
