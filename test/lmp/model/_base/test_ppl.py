r"""Test perplexity calculation.

Test target:
- :py:meth:`lmp.model.BaseModel.ppl`.
"""

import pytest
import torch
from typing import Dict

from lmp.model._base import BaseModel


@pytest.fixture
def batch_prev_tkids() -> torch.Tensor:
    r"""Return simple tensor with shape ``(B, S) == (2, 3)``."""
    return torch.zeros((2, 3))


@pytest.fixture
def batch_next_tkids_prob() -> torch.Tensor:
    r"""Return simple tensor with shape ``(B, S, V) == (2, 3, 4)``."""
    return torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.1],
            [0.3, 0.4, 0.1, 0.2],
        ],
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ],
    ])


@pytest.fixture
def batch_next_tkids() -> torch.Tensor:
    r"""Return simple tensor with shape ``(B, S) == (2, 3)``."""
    return torch.tensor([
        [0, 1, 2],
        [1, 2, 3],
    ])


@pytest.fixture
def batch_ppl() -> torch.Tensor:
    r"""Batch perplexity expected answer."""
    return torch.tensor([
        - 1 / 3 * (
            torch.log(torch.tensor([0.1]))
            + torch.log(torch.tensor([0.3]))
            + torch.log(torch.tensor([0.1]))
        ),
        - torch.log(torch.tensor([0.25])),
    ]).exp()


@pytest.fixture
def subclss_model(batch_next_tkids_prob: torch.Tensor):
    r"""Simple ``BaseModel`` subclass."""
    class SubclssModel(BaseModel):
        r"""Only implement `pred`."""

        def forward(self, **kwargs):
            pass

        def loss_fn(self, **kwargs):
            pass

        def pred(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
            r"""Return simple tensor with shape ``(B, S, V)``."""
            # (B, S, V) == (2, 3, 4)
            return batch_next_tkids_prob

    return SubclssModel


def test_perplexity_calculate_correctly(
        batch_next_tkids: torch.Tensor,
        batch_ppl: torch.Tensor,
        batch_prev_tkids: torch.Tensor,
        subclss_model: BaseModel,
):
    r"""Test perplexity calcuation result."""

    assert subclss_model().ppl(
        batch_next_tkids=batch_next_tkids,
        batch_prev_tkids=batch_prev_tkids,
    ) == batch_ppl.mean().item()
