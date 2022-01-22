"""Test perplexity calculation.

Test target:
- :py:meth:`lmp.util.metric.ppl`.
"""

import pytest
import torch

import lmp.util.metric


@pytest.fixture
def batch_tkids() -> torch.Tensor:
  """Mock target token ids with shape ``(B, S) == (2, 3)``."""
  return torch.tensor([
    [0, 1, 2],
    [1, 3, 2],
  ])


@pytest.fixture
def batch_tkids_pd(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Mock token ids probability distribution with shape ``(B, S, V) == (2, 3, 4)``."""
  return torch.tensor(
    [
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
      ],
      [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
      ],
    ]
  )


@pytest.fixture
def batch_ppl(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  """Expect perplexity result."""
  return torch.tensor([1.0, 1.0])


def test_calculate_result(batch_ppl: torch.Tensor, batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> None:
  """Test perplexity calcuation result."""
  assert torch.all(
    torch.isclose(
      input=lmp.util.metric.ppl(batch_tkids=batch_tkids, batch_tkids_pd=batch_tkids_pd),
      other=batch_ppl,
    )
  )
