"""Test model construction.

Test target:
- :py:meth:`lmp.model._lstm_2000.LSTM2000.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._lstm_2000 import LSTM2000
from lmp.tknzr._base import PAD_TKID, BaseTknzr


def test_parameters(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
  lstm_2000: LSTM2000,
) -> None:
  """Must correctly construct parameters."""
  lstm_2000.params_init()
  inv_sqrt_dim = 1 / math.sqrt(max(d_emb, n_blk * d_blk))

  assert hasattr(lstm_2000, 'd_blk')
  assert lstm_2000.d_blk == d_blk

  assert hasattr(lstm_2000, 'd_hid')
  assert lstm_2000.d_hid == n_blk * d_blk

  assert hasattr(lstm_2000, 'n_blk')
  assert lstm_2000.n_blk == n_blk

  assert hasattr(lstm_2000, 'emb')
  assert isinstance(lstm_2000.emb, nn.Embedding)
  assert lstm_2000.emb.embedding_dim == d_emb
  assert lstm_2000.emb.num_embeddings == tknzr.vocab_size
  assert lstm_2000.emb.padding_idx == PAD_TKID
  assert torch.all((-inv_sqrt_dim <= lstm_2000.emb.weight) & (lstm_2000.emb.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_e2fg')
  assert isinstance(lstm_2000.fc_e2fg, nn.Sequential)
  assert len(lstm_2000.fc_e2fg) == 2
  assert isinstance(lstm_2000.fc_e2fg[0], nn.Dropout)
  assert math.isclose(lstm_2000.fc_e2fg[0].p, p_emb)
  assert isinstance(lstm_2000.fc_e2fg[1], nn.Linear)
  assert lstm_2000.fc_e2fg[1].weight.size() == torch.Size([n_blk, d_emb])
  assert lstm_2000.fc_e2fg[1].bias.size() == torch.Size([n_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2fg[1].weight) & (lstm_2000.fc_e2fg[1].weight <= inv_sqrt_dim))
  assert torch.all((0 <= lstm_2000.fc_e2fg[1].bias) & (lstm_2000.fc_e2fg[1].bias <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_e2ig')
  assert isinstance(lstm_2000.fc_e2ig, nn.Sequential)
  assert len(lstm_2000.fc_e2ig) == 2
  assert isinstance(lstm_2000.fc_e2ig[0], nn.Dropout)
  assert math.isclose(lstm_2000.fc_e2ig[0].p, p_emb)
  assert isinstance(lstm_2000.fc_e2ig[1], nn.Linear)
  assert lstm_2000.fc_e2ig[1].weight.size() == torch.Size([n_blk, d_emb])
  assert lstm_2000.fc_e2ig[1].bias.size() == torch.Size([n_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2ig[1].weight) & (lstm_2000.fc_e2ig[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2ig[1].bias) & (lstm_2000.fc_e2ig[1].bias <= 0))

  assert hasattr(lstm_2000, 'fc_e2og')
  assert isinstance(lstm_2000.fc_e2og, nn.Sequential)
  assert len(lstm_2000.fc_e2og) == 2
  assert isinstance(lstm_2000.fc_e2og[0], nn.Dropout)
  assert math.isclose(lstm_2000.fc_e2og[0].p, p_emb)
  assert isinstance(lstm_2000.fc_e2og[1], nn.Linear)
  assert lstm_2000.fc_e2og[1].weight.size() == torch.Size([n_blk, d_emb])
  assert lstm_2000.fc_e2og[1].bias.size() == torch.Size([n_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2og[1].weight) & (lstm_2000.fc_e2og[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2og[1].bias) & (lstm_2000.fc_e2og[1].bias <= 0))

  assert hasattr(lstm_2000, 'fc_e2mc_in')
  assert isinstance(lstm_2000.fc_e2mc_in, nn.Sequential)
  assert len(lstm_2000.fc_e2mc_in) == 2
  assert isinstance(lstm_2000.fc_e2mc_in[0], nn.Dropout)
  assert math.isclose(lstm_2000.fc_e2mc_in[0].p, p_emb)
  assert isinstance(lstm_2000.fc_e2mc_in[1], nn.Linear)
  assert lstm_2000.fc_e2mc_in[1].weight.size() == torch.Size([n_blk * d_blk, d_emb])
  assert lstm_2000.fc_e2mc_in[1].bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2mc_in[1].weight) & (lstm_2000.fc_e2mc_in[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_e2mc_in[1].bias) & (lstm_2000.fc_e2mc_in[1].bias <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_h2fg')
  assert isinstance(lstm_2000.fc_h2fg, nn.Linear)
  assert lstm_2000.fc_h2fg.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_2000.fc_h2fg.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_h2fg.weight) & (lstm_2000.fc_h2fg.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_h2ig')
  assert isinstance(lstm_2000.fc_h2ig, nn.Linear)
  assert lstm_2000.fc_h2ig.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_2000.fc_h2ig.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_h2ig.weight) & (lstm_2000.fc_h2ig.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_h2og')
  assert isinstance(lstm_2000.fc_h2og, nn.Linear)
  assert lstm_2000.fc_h2og.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_2000.fc_h2og.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_h2og.weight) & (lstm_2000.fc_h2og.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_h2mc_in')
  assert isinstance(lstm_2000.fc_h2mc_in, nn.Linear)
  assert lstm_2000.fc_h2mc_in.weight.size() == torch.Size([n_blk * d_blk, n_blk * d_blk])
  assert lstm_2000.fc_h2mc_in.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_h2mc_in.weight) & (lstm_2000.fc_h2mc_in.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'h_0')
  assert isinstance(lstm_2000.h_0, nn.Parameter)
  assert lstm_2000.h_0.size() == torch.Size([1, n_blk * d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.h_0) & (lstm_2000.h_0 <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'c_0')
  assert isinstance(lstm_2000.c_0, nn.Parameter)
  assert lstm_2000.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.c_0) & (lstm_2000.c_0 <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'fc_h2e')
  assert isinstance(lstm_2000.fc_h2e, nn.Sequential)
  assert len(lstm_2000.fc_h2e) == 4
  assert isinstance(lstm_2000.fc_h2e[0], nn.Dropout)
  assert math.isclose(lstm_2000.fc_h2e[0].p, p_hid)
  assert lstm_2000.fc_h2e[1].weight.size() == torch.Size([d_emb, n_blk * d_blk])
  assert lstm_2000.fc_h2e[1].bias.size() == torch.Size([d_emb])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_h2e[1].weight) & (lstm_2000.fc_h2e[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2000.fc_h2e[1].bias) & (lstm_2000.fc_h2e[1].bias <= inv_sqrt_dim))
  assert isinstance(lstm_2000.fc_h2e[2], nn.Tanh)
  assert isinstance(lstm_2000.fc_h2e[3], nn.Dropout)
  assert math.isclose(lstm_2000.fc_h2e[3].p, p_hid)
