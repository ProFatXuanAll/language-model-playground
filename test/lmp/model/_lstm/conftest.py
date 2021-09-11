r"""Setup fixtures for testing :py:class:`lmp.model.LSTMModel`."""

import pytest
import torch

from lmp.model import LSTMModel
from lmp.tknzr import BaseTknzr


@pytest.fixture
def lstm_model(
        tknzr: BaseTknzr,
        d_emb: int,
        d_hid: int,
        n_hid_lyr: int,
        n_pre_hid_lyr: int,
        n_post_hid_lyr: int,
        p_emb: float,
        p_hid: float,
) -> LSTMModel:
    r"""Example ``LSTMModel`` instance."""
    return LSTMModel(
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        p_emb=p_emb,
        p_hid=p_hid,
        tknzr=tknzr,
    )


@pytest.fixture
def batch_prev_tkids(lstm_model: LSTMModel) -> torch.Tensor:
    r"""Example input batch of token ids."""
    # Shape: (2, 3).
    return torch.randint(
        low=0,
        high=lstm_model.emb.num_embeddings,
        size=(2, 3),
    )


@pytest.fixture
def batch_next_tkids(
    lstm_model: LSTMModel,
    batch_prev_tkids: torch.Tensor,
) -> torch.Tensor:
    r"""Example target batch of token ids."""
    # Same shape as `batch_prev_tkids`.
    return torch.cat(
        [
            batch_prev_tkids[..., :-1],
            torch.randint(
                low=0,
                high=lstm_model.emb.num_embeddings,
                size=(batch_prev_tkids.shape[0], 1),
            ),
        ],
        dim=1,
    )
