"""Setup fixtures for testing :py:class:`lmp.model._elman_net.ElmanNet`."""

import pytest
import torch

from lmp.model._elman_net import ElmanNet
from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def elman_net(d_emb: int, d_hid: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> ElmanNet:
  """:py:class:`lmp.model.ElmanNet` instance."""
  return ElmanNet(d_emb=d_emb, d_hid=d_hid, p_emb=p_emb, p_hid=p_hid, tknzr=tknzr)


@pytest.fixture
def batch_tkids(elman_net: ElmanNet) -> torch.Tensor:
  """Batch of token ids."""
  # Shape: (2, 4).
  return torch.randint(low=0, high=elman_net.emb.num_embeddings, size=(2, 4))


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
