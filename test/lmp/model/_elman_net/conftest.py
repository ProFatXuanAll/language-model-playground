"""Setup fixtures for testing :py:class:`lmp.model._elman_net`."""

import pytest
import torch

from lmp.model._elman_net import ElmanNet, ElmanNetLayer
from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def elman_net(
  d_emb: int,
  d_hid: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
) -> ElmanNet:
  """:py:class:`lmp.model._elman_net.ElmanNet` instance."""
  return ElmanNet(
    d_emb=d_emb,
    d_hid=d_hid,
    init_lower=init_lower,
    init_upper=init_upper,
    label_smoothing=label_smoothing,
    n_lyr=n_lyr,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )


@pytest.fixture
def elman_net_layer(
  in_feat: int,
  init_lower: float,
  init_upper: float,
  out_feat: int,
) -> ElmanNetLayer:
  """:py:class:`lmp.model._elman_net.ElmanNetLayer` instance."""
  return ElmanNetLayer(
    in_feat=in_feat,
    init_lower=init_lower,
    init_upper=init_upper,
    out_feat=out_feat,
  )


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


@pytest.fixture
def x(elman_net_layer: ElmanNetLayer) -> torch.Tensor:
  """Batch of input features."""
  # Shape: (2, 3, in_feat)
  return torch.rand((2, 3, elman_net_layer.in_feat))
