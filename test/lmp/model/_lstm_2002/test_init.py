"""Test model construction.

Test target:
- :py:meth:`lmp.model._lstm_2002.LSTM2002.__init__`.
- :py:meth:`lmp.model._lstm_2002.LSTM2002Layer.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._lstm_2002 import LSTM2002, LSTM2002Layer
from lmp.tknzr._base import PAD_TKID, BaseTknzr


def test_lstm_2002_layer_parameters(
  d_blk: int,
  in_feat: int,
  lstm_2002_layer: LSTM2002Layer,
  n_blk: int,
) -> None:
  """Must correctly construct parameters."""
  lstm_2002_layer.params_init()
  inv_sqrt_dim = 1 / math.sqrt(max(in_feat, n_blk * d_blk))

  assert lstm_2002_layer.d_blk == d_blk
  assert lstm_2002_layer.d_hid == n_blk * d_blk
  assert lstm_2002_layer.in_feat == in_feat
  assert lstm_2002_layer.n_blk == n_blk

  assert hasattr(lstm_2002_layer, 'fc_x2fg')
  assert isinstance(lstm_2002_layer.fc_x2fg, nn.Linear)
  assert lstm_2002_layer.fc_x2fg.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_2002_layer.fc_x2fg.bias.size() == torch.Size([n_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_x2fg.weight) & (lstm_2002_layer.fc_x2fg.weight <= inv_sqrt_dim))
  assert torch.all((0 <= lstm_2002_layer.fc_x2fg.bias) & (lstm_2002_layer.fc_x2fg.bias <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'fc_x2ig')
  assert isinstance(lstm_2002_layer.fc_x2ig, nn.Linear)
  assert lstm_2002_layer.fc_x2ig.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_2002_layer.fc_x2ig.bias.size() == torch.Size([n_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_x2ig.weight) & (lstm_2002_layer.fc_x2ig.weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_x2ig.bias) & (lstm_2002_layer.fc_x2ig.bias <= 0))

  assert hasattr(lstm_2002_layer, 'fc_x2og')
  assert isinstance(lstm_2002_layer.fc_x2og, nn.Linear)
  assert lstm_2002_layer.fc_x2og.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_2002_layer.fc_x2og.bias.size() == torch.Size([n_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_x2og.weight) & (lstm_2002_layer.fc_x2og.weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_x2og.bias) & (lstm_2002_layer.fc_x2og.bias <= 0))

  assert hasattr(lstm_2002_layer, 'fc_x2mc_in')
  assert isinstance(lstm_2002_layer.fc_x2mc_in, nn.Linear)
  assert lstm_2002_layer.fc_x2mc_in.weight.size() == torch.Size([n_blk * d_blk, in_feat])
  assert lstm_2002_layer.fc_x2mc_in.bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all(
    (-inv_sqrt_dim <= lstm_2002_layer.fc_x2mc_in.weight) & (lstm_2002_layer.fc_x2mc_in.weight <= inv_sqrt_dim)
  )
  assert torch.all(
    (-inv_sqrt_dim <= lstm_2002_layer.fc_x2mc_in.bias) & (lstm_2002_layer.fc_x2mc_in.bias <= inv_sqrt_dim)
  )

  assert hasattr(lstm_2002_layer, 'fc_h2fg')
  assert isinstance(lstm_2002_layer.fc_h2fg, nn.Linear)
  assert lstm_2002_layer.fc_h2fg.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_2002_layer.fc_h2fg.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_h2fg.weight) & (lstm_2002_layer.fc_h2fg.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'fc_h2ig')
  assert isinstance(lstm_2002_layer.fc_h2ig, nn.Linear)
  assert lstm_2002_layer.fc_h2ig.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_2002_layer.fc_h2ig.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_h2ig.weight) & (lstm_2002_layer.fc_h2ig.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'fc_h2og')
  assert isinstance(lstm_2002_layer.fc_h2og, nn.Linear)
  assert lstm_2002_layer.fc_h2og.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_2002_layer.fc_h2og.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.fc_h2og.weight) & (lstm_2002_layer.fc_h2og.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'fc_h2mc_in')
  assert isinstance(lstm_2002_layer.fc_h2mc_in, nn.Linear)
  assert lstm_2002_layer.fc_h2mc_in.weight.size() == torch.Size([n_blk * d_blk, n_blk * d_blk])
  assert lstm_2002_layer.fc_h2mc_in.bias is None
  assert torch.all(
    (-inv_sqrt_dim <= lstm_2002_layer.fc_h2mc_in.weight) & (lstm_2002_layer.fc_h2mc_in.weight <= inv_sqrt_dim)
  )

  assert hasattr(lstm_2002_layer, 'h_0')
  assert isinstance(lstm_2002_layer.h_0, nn.Parameter)
  assert lstm_2002_layer.h_0.size() == torch.Size([1, n_blk * d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.h_0) & (lstm_2002_layer.h_0 <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'c_0')
  assert isinstance(lstm_2002_layer.c_0, nn.Parameter)
  assert lstm_2002_layer.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.c_0) & (lstm_2002_layer.c_0 <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'pc_c2fg')
  assert isinstance(lstm_2002_layer.pc_c2fg, nn.Parameter)
  assert lstm_2002_layer.pc_c2fg.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.pc_c2fg) & (lstm_2002_layer.pc_c2fg <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'pc_c2ig')
  assert isinstance(lstm_2002_layer.pc_c2ig, nn.Parameter)
  assert lstm_2002_layer.pc_c2ig.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.pc_c2ig) & (lstm_2002_layer.pc_c2ig <= inv_sqrt_dim))

  assert hasattr(lstm_2002_layer, 'pc_c2og')
  assert isinstance(lstm_2002_layer.pc_c2og, nn.Parameter)
  assert lstm_2002_layer.pc_c2og.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002_layer.pc_c2og) & (lstm_2002_layer.pc_c2og <= inv_sqrt_dim))


def test_lstm_2002_parameters(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
  lstm_2002: LSTM2002,
) -> None:
  """Must correctly construct parameters."""
  lstm_2002.params_init()
  inv_sqrt_dim = 1 / math.sqrt(max(d_emb, n_blk * d_blk))

  assert hasattr(lstm_2002, 'd_blk')
  assert lstm_2002.d_blk == d_blk

  assert hasattr(lstm_2002, 'd_hid')
  assert lstm_2002.d_hid == n_blk * d_blk

  assert hasattr(lstm_2002, 'n_blk')
  assert lstm_2002.n_blk == n_blk

  assert hasattr(lstm_2002, 'emb')
  assert isinstance(lstm_2002.emb, nn.Embedding)
  assert lstm_2002.emb.embedding_dim == d_emb
  assert lstm_2002.emb.num_embeddings == tknzr.vocab_size
  assert lstm_2002.emb.padding_idx == PAD_TKID
  assert torch.all((-inv_sqrt_dim <= lstm_2002.emb.weight) & (lstm_2002.emb.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2002, 'fc_e2h')
  assert isinstance(lstm_2002.fc_e2h, nn.Sequential)
  assert len(lstm_2002.fc_e2h) == 4
  assert isinstance(lstm_2002.fc_e2h[0], nn.Dropout)
  assert math.isclose(lstm_2002.fc_e2h[0].p, p_emb)
  assert isinstance(lstm_2002.fc_e2h[1], nn.Linear)
  assert lstm_2002.fc_e2h[1].weight.size() == torch.Size([n_blk * d_blk, d_emb])
  assert lstm_2002.fc_e2h[1].bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2002.fc_e2h[1].weight) & (lstm_2002.fc_e2h[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2002.fc_e2h[1].bias) & (lstm_2002.fc_e2h[1].bias <= inv_sqrt_dim))
  assert isinstance(lstm_2002.fc_e2h[2], nn.Tanh)
  assert isinstance(lstm_2002.fc_e2h[3], nn.Dropout)
  assert math.isclose(lstm_2002.fc_e2h[3].p, p_hid)

  assert hasattr(lstm_2002, 'stack_rnn')
  assert isinstance(lstm_2002.stack_rnn, nn.ModuleList)
  assert len(lstm_2002.stack_rnn) == 2 * n_lyr
  for lyr in range(n_lyr):
    rnn_lyr = lstm_2002.stack_rnn[2 * lyr]
    assert isinstance(rnn_lyr, LSTM2002Layer)
    assert rnn_lyr.d_blk == d_blk
    assert rnn_lyr.in_feat == n_blk * d_blk
    assert rnn_lyr.n_blk == n_blk

    dropout_lyr = lstm_2002.stack_rnn[2 * lyr + 1]
    assert isinstance(dropout_lyr, nn.Dropout)
    assert math.isclose(dropout_lyr.p, p_hid)

  assert hasattr(lstm_2002, 'fc_h2e')
  assert isinstance(lstm_2002.fc_h2e, nn.Sequential)
  assert len(lstm_2002.fc_h2e) == 3
  assert isinstance(lstm_2002.fc_h2e[0], nn.Linear)
  assert lstm_2002.fc_h2e[0].weight.size() == torch.Size([d_emb, n_blk * d_blk])
  assert lstm_2002.fc_h2e[0].bias.size() == torch.Size([d_emb])
  assert torch.all((-inv_sqrt_dim <= lstm_2002.fc_h2e[0].weight) & (lstm_2002.fc_h2e[0].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2002.fc_h2e[0].bias) & (lstm_2002.fc_h2e[0].bias <= inv_sqrt_dim))
  assert isinstance(lstm_2002.fc_h2e[1], nn.Tanh)
  assert isinstance(lstm_2002.fc_h2e[2], nn.Dropout)
  assert math.isclose(lstm_2002.fc_h2e[2].p, p_hid)
