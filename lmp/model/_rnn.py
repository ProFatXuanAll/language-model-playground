r"""Vanilla RNN language model."""

import argparse
from typing import ClassVar, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


class RNNModel(BaseModel):
    r"""Vanilla RNN language model.

    Use ``self.loss_fn`` for training and use ``self.pred`` for inference.
    Both are depended on forward pass alogorithm ``self.forward``.

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
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of RNN layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_lyr: int
        Number of MLP layers ``+1`` after RNN layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_post_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    n_pre_hid_lyr: int
        Number of MLP layers ``+1`` before RNN layer.
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
    hid: torch.nn.RNN
        Vanilla RNN which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Dropout temporal features with probability ``p_hid`` if
        ``n_hid_lyr > 1``.
    model_name: ClassVar[str]
        Model name is ``RNN``.
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
    model_name: ClassVar[str] = 'RNN'

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
        super().__init__()

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
            padding_idx=tknzr.pad_tkid,
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

        # Rectified MLP which transform token embeddings from embedding
        # dimension `d_emb` to hidden dimension `d_hid`.
        # Drop rectified units with probability `p_hid`.
        # Input tensor : Output of `self.emb_dp`.
        # Input shape  : `(B, S, E)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of sparse rectified token representation.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        pre_hid: List[nn.Module] = [
            nn.Linear(in_features=d_emb, out_features=d_hid),
            nn.ReLU(),
            nn.Dropout(p=p_hid),
        ]

        for _ in range(n_pre_hid_lyr):
            pre_hid.append(nn.Linear(in_features=d_hid, out_features=d_hid))
            pre_hid.append(nn.ReLU())
            pre_hid.append(nn.Dropout(p=p_hid))

        self.pre_hid = nn.Sequential(*pre_hid)

        # Vanilla RNN which encode temporal features.
        # Each time step's hidden state depends on current input and previous
        # hidden state.
        # Dropout temporal features if `n_hid_lyr > 1`.
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.hid: nn.Module
        if n_hid_lyr == 1:
            self.hid = nn.RNN(
                input_size=d_hid,
                hidden_size=d_hid,
                batch_first=True,
            )
        else:
            self.hid = nn.RNN(
                input_size=d_hid,
                hidden_size=d_hid,
                num_layers=n_hid_lyr,
                dropout=p_hid,
                batch_first=True,
            )

        # Rectified MLP which transform temporal features from hidden dimension
        # `d_hid` to embedding dimension `d_emb`.
        # Drop rectified units with probability `p_hid`.
        # Input tensor : Output of `self.hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of sparse rectified next token embeddings.
        # Output shape : `(B, S, E)`.
        # Output dtype : `torch.float32`.
        post_hid: List[nn.Module] = []
        for _ in range(n_post_hid_lyr):
            post_hid.append(nn.Dropout(p=p_hid))
            post_hid.append(nn.Linear(in_features=d_hid, out_features=d_hid))
            post_hid.append(nn.ReLU())

        post_hid.append(nn.Dropout(p=p_hid))
        post_hid.append(nn.Linear(in_features=d_hid, out_features=d_emb))
        self.post_hid = nn.Sequential(*post_hid)

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
        batch, _ = self.hid(batch)

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
        r"""Training vanilla RNN language model CLI arguments parser.

        Parameters
        ==========
        parser: argparse.ArgumentParser
            Parser for CLI arguments.

        See Also
        ========
        lmp.model.BaseModel.train_parser
            Training language model CLI arguments.
        lmp.script.train_model
            Language model training script.

        Examples
        ========
        >>> import argparse
        >>> from lmp.model import RNNModel
        >>> parser = argparse.ArgumentParser()
        >>> RNNModel.train_parser(parser)
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
        ...     '--n_epoch', '10',
        ...     '--tknzr_exp_name', 'my_tknzr_exp',
        ...     '--ver', 'train',
        ...     '--wd', '1e-2',
        ...     '--d_emb', '100',
        ...     '--d_hid', '300',
        ...     '--n_hid_lyr', '2',
        ...     '--n_post_hid_lyr', '1',
        ...     '--n_pre_hid_lyr', '1',
        ...     '--p_emb', '0.1',
        ...     '--p_hid', '0.1',
        ... ])
        >>> args.d_emb == 100
        True
        >>> args.d_hid == 300
        True
        >>> args.n_hid_lyr == 2
        True
        >>> args.n_post_hid_lyr == 1
        True
        >>> args.n_pre_hid_lyr == 1
        True
        >>> args.p_emb == 0.1
        True
        >>> args.p_hid == 0.1
        True
        """
        # Load common arguments.
        BaseModel.train_parser(parser=parser)

        # Required arguments.
        group = parser.add_argument_group('model arguments')
        group.add_argument(
            '--d_emb',
            help='Token embedding dimension.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--d_hid',
            help='Hidden dimension for MLP and RNN.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--n_hid_lyr',
            help='Number of RNN layers.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--n_post_hid_lyr',
            help='Number of MLP layers ``+ 1`` after RNN layer.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--n_pre_hid_lyr',
            help='Number of MLP layers ``+ 1`` before RNN layer.',
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
            help='Dropout probability for hidden representation.',
            required=True,
            type=float,
        )
