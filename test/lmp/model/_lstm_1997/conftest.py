"""Setup fixtures for testing :py:class:`lmp.model._lstm_1997.LSTM1997`."""

import pytest
import torch

from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer
from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def lstm_1997(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
) -> LSTM1997:
  """:py:class:`lmp.model._lstm_1997.LSTM1997` instance."""
  return LSTM1997(d_blk=d_blk, d_emb=d_emb, n_blk=n_blk, n_lyr=n_lyr, p_emb=p_emb, p_hid=p_hid, tknzr=tknzr)


@pytest.fixture
def lstm_1997_layer(d_blk: int, in_feat: int, n_blk: int) -> LSTM1997Layer:
  """:py:class:`lmp.model._lstm_1997.LSTM1997Layer` instance."""
  return LSTM1997Layer(d_blk=d_blk, in_feat=in_feat, n_blk=n_blk)


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


@pytest.fixture
def batch_x(lstm_1997_layer: LSTM1997Layer) -> torch.Tensor:
  """Batch of input features."""
  # Shape: (2, 3, in_feat)
  return torch.rand((2, 3, lstm_1997_layer.in_feat))
