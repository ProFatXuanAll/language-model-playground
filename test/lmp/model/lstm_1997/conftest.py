"""Setup fixtures for testing :py:class:`lmp.model.LSTM1997`."""

import pytest
import torch

from lmp.model import LSTM1997
from lmp.tknzr import BaseTknzr


@pytest.fixture
def lstm_1997(d_cell: int, d_emb: int, n_cell: int, tknzr: BaseTknzr) -> LSTM1997:
  """:py:class:`lmp.model.LSTM1997` instance."""
  return LSTM1997(d_cell=d_cell, d_emb=d_emb, n_cell=n_cell, tknzr=tknzr)


@pytest.fixture
def batch_tkids(lstm_1997: LSTM1997) -> torch.Tensor:
  """Batch of token ids."""
  # Shape: (2, 4).
  return torch.randint(low=0, high=lstm_1997.emb.num_embeddings, size=(2, 4))


@pytest.fixture
def batch_cur_tkids(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Batch of input token ids."""
  # Shape: (2, 3).
  return batch_tkids[..., :-1]


@pytest.fixture
def batch_next_tkids(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Batch of target token ids."""
  # Shape: (2, 3).
  return batch_tkids[..., 1:]
