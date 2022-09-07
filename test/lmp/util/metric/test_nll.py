"""Test -log(p) calculation.

Test target:
- :py:meth:`lmp.util.metric.nll`.
"""

import pytest
import torch

import lmp.util.metric
from lmp.vars import BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID


@pytest.fixture
def batch_tkids() -> torch.Tensor:
  """Mock target token ids with shape ``(B, S) == (2, 5)``."""
  return torch.tensor(
    [
      [BOS_TKID, UNK_TKID, EOS_TKID, PAD_TKID, PAD_TKID],
      [BOS_TKID, UNK_TKID, UNK_TKID, UNK_TKID, EOS_TKID],
    ]
  )


@pytest.fixture
def batch_tkids_pd(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Mock token ids probability distribution with shape ``(B, S, V) == (2, 5, 4)``."""
  pd = torch.zeros(2, 5, 4)
  pd[0, 0, BOS_TKID] = 0.9
  pd[0, 1, UNK_TKID] = 0.8
  pd[0, 2, EOS_TKID] = 0.7
  pd[0, 3, PAD_TKID] = 0.6
  pd[0, 4, PAD_TKID] = 0.5
  pd[1, 0, BOS_TKID] = 0.4
  pd[1, 1, UNK_TKID] = 0.3
  pd[1, 2, UNK_TKID] = 0.2
  pd[1, 3, UNK_TKID] = 0.1
  pd[1, 4, EOS_TKID] = 0.1
  return pd


@pytest.fixture
def batch_nll(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  """Expect perplexity result.

  Must has shape ``(B) == (2)``.
  """
  # Paddings are masked.
  p = torch.tensor([
    [0.9, 0.8, 0.7, 1.0, 1.0],
    [0.4, 0.3, 0.2, 0.1, 0.1],
  ])
  return -p.log2()


def test_calculate_result(batch_nll: torch.Tensor, batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> None:
  """Test perplexity calcuation result."""
  assert torch.all(
    torch.isclose(
      input=lmp.util.metric.nll(
        batch_tkids=batch_tkids,
        batch_tkids_pd=batch_tkids_pd,
      ),
      other=batch_nll,
    )
  )
