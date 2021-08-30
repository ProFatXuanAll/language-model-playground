r"""LSTM language model."""

from typing import ClassVar, Dict, Optional

import torch.nn as nn

from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


class LSTMModel(RNNModel):
    r"""LSTM language model.


    Same architecture as :py:class:`lmp.model.RNNModel` but use LSTM layer.

    Parameters
    ==========
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_hid: int
        Hidden dimension for MLP and LSTM.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of LSTM layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_lyr: int
        Number of MLP layers ``+1`` after LSTM layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_post_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    n_pre_hid_lyr: int
        Number of MLP layers ``+1`` before LSTM layer.
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
    hid: torch.nn.LSTM
        LSTM which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Dropout temporal features with probability ``p_hid`` if
        ``n_hid_lyr > 1``.
    model_name: ClassVar[str]
        Model name is ``LSTM``.
        Used for command line argument parsing.
    """
    model_name: ClassVar[str] = 'LSTM'

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

        # Override RNN layer with LSTM layer.
        # Dropout temporal features if `n_hid_lyr > 1`.
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        if n_hid_lyr == 1:
            self.hid = nn.LSTM(
                input_size=d_hid,
                hidden_size=d_hid,
                batch_first=True
            )
        else:
            self.hid = nn.LSTM(
                input_size=d_hid,
                hidden_size=d_hid,
                num_layers=n_hid_lyr,
                dropout=p_hid,
                batch_first=True
            )
