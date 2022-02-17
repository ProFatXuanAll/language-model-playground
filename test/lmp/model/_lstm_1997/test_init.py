"""Test model construction.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._lstm_1997 import LSTM1997
from lmp.tknzr._base import PAD_TKID, BaseTknzr


def test_parameters(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
  lstm_1997: LSTM1997,
) -> None:
  """Must correctly construct parameters."""
  inv_sqrt_dim = 1 / math.sqrt(max(d_emb, n_blk * d_blk))

  assert hasattr(lstm_1997, 'd_hid')
  assert lstm_1997.d_hid == n_blk * d_blk

  assert hasattr(lstm_1997, 'ig_range')
  assert lstm_1997.ig_range == (0, n_blk)

  assert hasattr(lstm_1997, 'og_range')
  assert lstm_1997.og_range == (n_blk, 2 * n_blk)

  assert hasattr(lstm_1997, 'g_range')
  assert lstm_1997.g_range == (0, 2 * n_blk)

  assert hasattr(lstm_1997, 'mc_range')
  assert lstm_1997.mc_range == (2 * n_blk, n_blk * (2 + d_blk))

  assert hasattr(lstm_1997, 'emb')
  assert isinstance(lstm_1997.emb, nn.Embedding)
  assert lstm_1997.emb.embedding_dim == d_emb
  assert lstm_1997.emb.num_embeddings == tknzr.vocab_size
  assert lstm_1997.emb.padding_idx == PAD_TKID
  assert torch.all((-inv_sqrt_dim <= lstm_1997.emb.weight) & (lstm_1997.emb.weight <= inv_sqrt_dim))

  assert hasattr(lstm_1997, 'proj_e2cg')
  assert isinstance(lstm_1997.proj_e2cg, nn.Sequential)
  assert len(lstm_1997.proj_e2cg) == 2
  assert isinstance(lstm_1997.proj_e2cg[0], nn.Dropout)
  assert math.isclose(lstm_1997.proj_e2cg[0].p, p_emb)
  assert isinstance(lstm_1997.proj_e2cg[1], nn.Linear)
  assert lstm_1997.proj_e2cg[1].weight.size() == torch.Size([n_blk * (2 + d_blk), d_emb])
  assert lstm_1997.proj_e2cg[1].bias.size() == torch.Size([n_blk * (2 + d_blk)])
  assert torch.all((-inv_sqrt_dim <= lstm_1997.proj_e2cg[1].weight) & (lstm_1997.proj_e2cg[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_1997.proj_e2cg[1].bias) & (lstm_1997.proj_e2cg[1].bias <= inv_sqrt_dim))
  assert torch.all(
    (-inv_sqrt_dim <= lstm_1997.proj_e2cg[1].bias[lstm_1997.g_range[0]:lstm_1997.g_range[1]]) &
    (lstm_1997.proj_e2cg[1].bias[lstm_1997.g_range[0]:lstm_1997.g_range[1]] <= 0)
  )

  assert hasattr(lstm_1997, 'proj_h2cg')
  assert isinstance(lstm_1997.proj_h2cg, nn.Linear)
  assert lstm_1997.proj_h2cg.weight.size() == torch.Size([n_blk * (2 + d_blk), n_blk * d_blk])
  assert lstm_1997.proj_h2cg.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_1997.proj_h2cg.weight) & (lstm_1997.proj_h2cg.weight <= inv_sqrt_dim))

  assert hasattr(lstm_1997, 'h_0')
  assert isinstance(lstm_1997.h_0, nn.Parameter)
  assert lstm_1997.h_0.size() == torch.Size([1, n_blk * d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_1997.h_0) & (lstm_1997.h_0 <= inv_sqrt_dim))

  assert hasattr(lstm_1997, 'c_0')
  assert isinstance(lstm_1997.c_0, nn.Parameter)
  assert lstm_1997.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_1997.c_0) & (lstm_1997.c_0 <= inv_sqrt_dim))

  assert hasattr(lstm_1997, 'proj_h2e')
  assert isinstance(lstm_1997.proj_h2e, nn.Sequential)
  assert len(lstm_1997.proj_h2e) == 4
  assert isinstance(lstm_1997.proj_h2e[0], nn.Dropout)
  assert math.isclose(lstm_1997.proj_h2e[0].p, p_hid)
  assert lstm_1997.proj_h2e[1].weight.size() == torch.Size([d_emb, n_blk * d_blk])
  assert lstm_1997.proj_h2e[1].bias.size() == torch.Size([d_emb])
  assert torch.all((-inv_sqrt_dim <= lstm_1997.proj_h2e[1].weight) & (lstm_1997.proj_h2e[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_1997.proj_h2e[1].bias) & (lstm_1997.proj_h2e[1].bias <= inv_sqrt_dim))
  assert isinstance(lstm_1997.proj_h2e[2], nn.Tanh)
  assert isinstance(lstm_1997.proj_h2e[3], nn.Dropout)
  assert math.isclose(lstm_1997.proj_h2e[3].p, p_hid)
