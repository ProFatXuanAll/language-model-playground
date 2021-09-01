r"""Setup fixtures for testing :py:class:`lmp.model.GRUModel`."""

import pytest
import torch

from lmp.model import GRUModel


@pytest.fixture
def gru_model(
        tknzr,
        d_emb,
        d_hid,
        n_hid_lyr,
        n_pre_hid_lyr,
        n_post_hid_lyr,
        p_emb,
        p_hid,
) -> GRUModel:
    r"""Example GRUModel instance."""
    return GRUModel(
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
def batch_prev_tkids(gru_model: GRUModel) -> torch.Tensor:
    r"""Example input batch of token ids."""
    # Shape: (2, 3).
    return torch.randint(
        low=0,
        high=gru_model.emb.num_embeddings,
        size=(2, 3),
    )


@pytest.fixture
def batch_next_tkids(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
) -> torch.Tensor:
    r"""Example target batch of token ids."""
    # Same shape as `batch_prev_tkids`.
    return torch.cat(
        [
            batch_prev_tkids[..., :-1],
            torch.randint(
                low=0,
                high=gru_model.emb.num_embeddings,
                size=(batch_prev_tkids.shape[0], 1),
            ),
        ],
        dim=1,
    )
