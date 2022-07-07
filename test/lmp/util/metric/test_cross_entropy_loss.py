"""Test cross entropy loss calculation.

Test target:
- :py:meth:`lmp.util.metric.cross_entropy_loss`.
"""

import math

import pytest
import torch

import lmp.util.metric
from lmp.tknzr._base import BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID


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
def batch_cross_entropy_loss(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  """Expect cross entropy loss result.

  Must has shape ``(1)``.
  """
  # Paddings are masked.
  log_p = math.log(0.9) + math.log(0.8) + math.log(0.7)
  log_p = log_p + math.log(0.4) + math.log(0.3) + math.log(0.2) + math.log(0.1) + math.log(0.1)
  log_p = log_p / 8
  return torch.tensor(-log_p)


def test_calculate_result(
  batch_cross_entropy_loss: torch.Tensor,
  batch_tkids: torch.Tensor,
  batch_tkids_pd: torch.Tensor,
) -> None:
  """Test cross entropy loss calcuation result."""
  assert torch.isclose(
    input=lmp.util.metric.cross_entropy_loss(
      batch_tkids=batch_tkids,
      batch_tkids_pd=batch_tkids_pd,
    ),
    other=batch_cross_entropy_loss,
  )
