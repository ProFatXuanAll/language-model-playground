r"""Setup fixtures for testing :py:class:`lmp.model.RNNModel`."""

import pytest
import torch

from lmp.model import RNNModel


@pytest.fixture
def rnn_model(tknzr) -> RNNModel:
    r"""Example RNNModel instance."""
    return RNNModel(
        d_emb=1,
        d_hid=1,
        n_hid_lyr=1,
        n_pre_hid_lyr=1,
        n_post_hid_lyr=1,
        p_emb=0.5,
        p_hid=0.5,
        tknzr=tknzr,
    )


@pytest.fixture
def batch_prev_tkids(rnn_model: RNNModel) -> torch.Tensor:
    r"""Example input batch of token ids."""
    # Shape: (2, 3).
    return torch.randint(
        low=0,
        high=rnn_model.emb.num_embeddings,
        size=(2, 3),
    )


@pytest.fixture
def batch_next_tkids(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
) -> torch.Tensor:
    r"""Example target batch of token ids."""
    # Same shape as `batch_prev_tkids`.
    return torch.cat(
        [
            batch_prev_tkids[..., :-1],
            torch.randint(
                low=0,
                high=rnn_model.emb.num_embeddings,
                size=(batch_prev_tkids.shape[0], 1),
            ),
        ],
        dim=1,
    )
