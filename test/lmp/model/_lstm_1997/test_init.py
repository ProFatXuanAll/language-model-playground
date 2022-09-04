"""Test model construction.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.__init__`.
- :py:meth:`lmp.model._lstm_1997.LSTM1997Layer.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer
from lmp.tknzr._base import BaseTknzr
from lmp.vars import PAD_TKID


def test_lstm_1997_layer_default_value() -> None:
  """Ensure default value consistency."""
  in_feat = 1
  d_blk = 1
  init_ib = -1.0
  init_lower = -0.1
  init_ob = -1.0
  init_upper = 0.1
  n_blk = 1
  lstm_1997_layer = LSTM1997Layer()
  lstm_1997_layer.params_init()

  assert hasattr(lstm_1997_layer, 'd_blk')
  assert lstm_1997_layer.d_blk == d_blk

  assert hasattr(lstm_1997_layer, 'd_hid')
  assert lstm_1997_layer.d_hid == n_blk * d_blk

  assert hasattr(lstm_1997_layer, 'in_feat')
  assert lstm_1997_layer.in_feat == in_feat

  assert hasattr(lstm_1997_layer, 'init_ib')
  assert math.isclose(lstm_1997_layer.init_ib, init_ib)

  assert hasattr(lstm_1997_layer, 'init_lower')
  assert math.isclose(lstm_1997_layer.init_lower, init_lower)

  assert hasattr(lstm_1997_layer, 'init_ob')
  assert math.isclose(lstm_1997_layer.init_ob, init_ob)

  assert hasattr(lstm_1997_layer, 'init_upper')
  assert math.isclose(lstm_1997_layer.init_upper, init_upper)

  assert hasattr(lstm_1997_layer, 'n_blk')
  assert lstm_1997_layer.n_blk == n_blk

  assert hasattr(lstm_1997_layer, 'fc_x2ig')
  assert isinstance(lstm_1997_layer.fc_x2ig, nn.Linear)
  assert lstm_1997_layer.fc_x2ig.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_1997_layer.fc_x2ig.bias.size() == torch.Size([n_blk])
  assert torch.all((init_lower <= lstm_1997_layer.fc_x2ig.weight) & (lstm_1997_layer.fc_x2ig.weight <= init_upper))
  assert torch.all((init_ib <= lstm_1997_layer.fc_x2ig.bias) & (lstm_1997_layer.fc_x2ig.bias <= 0))

  assert hasattr(lstm_1997_layer, 'fc_x2og')
  assert isinstance(lstm_1997_layer.fc_x2og, nn.Linear)
  assert lstm_1997_layer.fc_x2og.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_1997_layer.fc_x2og.bias.size() == torch.Size([n_blk])
  assert torch.all((init_lower <= lstm_1997_layer.fc_x2og.weight) & (lstm_1997_layer.fc_x2og.weight <= init_upper))
  assert torch.all((init_ob <= lstm_1997_layer.fc_x2og.bias) & (lstm_1997_layer.fc_x2og.bias <= 0))

  assert hasattr(lstm_1997_layer, 'fc_x2mc_in')
  assert isinstance(lstm_1997_layer.fc_x2mc_in, nn.Linear)
  assert lstm_1997_layer.fc_x2mc_in.weight.size() == torch.Size([n_blk * d_blk, in_feat])
  assert lstm_1997_layer.fc_x2mc_in.bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all(
    (init_lower <= lstm_1997_layer.fc_x2mc_in.weight) & (lstm_1997_layer.fc_x2mc_in.weight <= init_upper)
  )
  assert torch.all((init_lower <= lstm_1997_layer.fc_x2mc_in.bias) & (lstm_1997_layer.fc_x2mc_in.bias <= init_upper))

  assert hasattr(lstm_1997_layer, 'fc_h2ig')
  assert isinstance(lstm_1997_layer.fc_h2ig, nn.Linear)
  assert lstm_1997_layer.fc_h2ig.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_1997_layer.fc_h2ig.bias is None
  assert torch.all((init_lower <= lstm_1997_layer.fc_h2ig.weight) & (lstm_1997_layer.fc_h2ig.weight <= init_upper))

  assert hasattr(lstm_1997_layer, 'fc_h2og')
  assert isinstance(lstm_1997_layer.fc_h2og, nn.Linear)
  assert lstm_1997_layer.fc_h2og.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_1997_layer.fc_h2og.bias is None
  assert torch.all((init_lower <= lstm_1997_layer.fc_h2og.weight) & (lstm_1997_layer.fc_h2og.weight <= init_upper))

  assert hasattr(lstm_1997_layer, 'fc_h2mc_in')
  assert isinstance(lstm_1997_layer.fc_h2mc_in, nn.Linear)
  assert lstm_1997_layer.fc_h2mc_in.weight.size() == torch.Size([n_blk * d_blk, n_blk * d_blk])
  assert lstm_1997_layer.fc_h2mc_in.bias is None
  assert torch.all(
    (init_lower <= lstm_1997_layer.fc_h2mc_in.weight) & (lstm_1997_layer.fc_h2mc_in.weight <= init_upper)
  )

  assert hasattr(lstm_1997_layer, 'c_0')
  assert isinstance(lstm_1997_layer.c_0, torch.Tensor)
  assert lstm_1997_layer.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all(lstm_1997_layer.c_0 == 0.0)

  assert hasattr(lstm_1997_layer, 'h_0')
  assert isinstance(lstm_1997_layer.h_0, torch.Tensor)
  assert lstm_1997_layer.h_0.size() == torch.Size([1, n_blk * d_blk])
  assert torch.all(lstm_1997_layer.h_0 == 0.0)


def test_lstm_1997_layer_parameters(
  d_blk: int,
  in_feat: int,
  init_ib: float,
  init_lower: float,
  init_ob: float,
  init_upper: float,
  lstm_1997_layer: LSTM1997Layer,
  n_blk: int,
) -> None:
  """Must correctly construct parameters."""
  lstm_1997_layer.params_init()

  assert hasattr(lstm_1997_layer, 'd_blk')
  assert lstm_1997_layer.d_blk == d_blk

  assert hasattr(lstm_1997_layer, 'd_hid')
  assert lstm_1997_layer.d_hid == n_blk * d_blk

  assert hasattr(lstm_1997_layer, 'in_feat')
  assert lstm_1997_layer.in_feat == in_feat

  assert hasattr(lstm_1997_layer, 'init_ib')
  assert math.isclose(lstm_1997_layer.init_ib, init_ib)

  assert hasattr(lstm_1997_layer, 'init_lower')
  assert math.isclose(lstm_1997_layer.init_lower, init_lower)

  assert hasattr(lstm_1997_layer, 'init_ob')
  assert math.isclose(lstm_1997_layer.init_ob, init_ob)

  assert hasattr(lstm_1997_layer, 'init_upper')
  assert math.isclose(lstm_1997_layer.init_upper, init_upper)

  assert hasattr(lstm_1997_layer, 'n_blk')
  assert lstm_1997_layer.n_blk == n_blk

  assert hasattr(lstm_1997_layer, 'fc_x2ig')
  assert isinstance(lstm_1997_layer.fc_x2ig, nn.Linear)
  assert lstm_1997_layer.fc_x2ig.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_1997_layer.fc_x2ig.bias.size() == torch.Size([n_blk])
  assert torch.all((init_lower <= lstm_1997_layer.fc_x2ig.weight) & (lstm_1997_layer.fc_x2ig.weight <= init_upper))
  assert torch.all((init_ib <= lstm_1997_layer.fc_x2ig.bias) & (lstm_1997_layer.fc_x2ig.bias <= 0))

  assert hasattr(lstm_1997_layer, 'fc_x2og')
  assert isinstance(lstm_1997_layer.fc_x2og, nn.Linear)
  assert lstm_1997_layer.fc_x2og.weight.size() == torch.Size([n_blk, in_feat])
  assert lstm_1997_layer.fc_x2og.bias.size() == torch.Size([n_blk])
  assert torch.all((init_lower <= lstm_1997_layer.fc_x2og.weight) & (lstm_1997_layer.fc_x2og.weight <= init_upper))
  assert torch.all((init_ob <= lstm_1997_layer.fc_x2og.bias) & (lstm_1997_layer.fc_x2og.bias <= 0))

  assert hasattr(lstm_1997_layer, 'fc_x2mc_in')
  assert isinstance(lstm_1997_layer.fc_x2mc_in, nn.Linear)
  assert lstm_1997_layer.fc_x2mc_in.weight.size() == torch.Size([n_blk * d_blk, in_feat])
  assert lstm_1997_layer.fc_x2mc_in.bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all(
    (init_lower <= lstm_1997_layer.fc_x2mc_in.weight) & (lstm_1997_layer.fc_x2mc_in.weight <= init_upper)
  )
  assert torch.all((init_lower <= lstm_1997_layer.fc_x2mc_in.bias) & (lstm_1997_layer.fc_x2mc_in.bias <= init_upper))

  assert hasattr(lstm_1997_layer, 'fc_h2ig')
  assert isinstance(lstm_1997_layer.fc_h2ig, nn.Linear)
  assert lstm_1997_layer.fc_h2ig.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_1997_layer.fc_h2ig.bias is None
  assert torch.all((init_lower <= lstm_1997_layer.fc_h2ig.weight) & (lstm_1997_layer.fc_h2ig.weight <= init_upper))

  assert hasattr(lstm_1997_layer, 'fc_h2og')
  assert isinstance(lstm_1997_layer.fc_h2og, nn.Linear)
  assert lstm_1997_layer.fc_h2og.weight.size() == torch.Size([n_blk, n_blk * d_blk])
  assert lstm_1997_layer.fc_h2og.bias is None
  assert torch.all((init_lower <= lstm_1997_layer.fc_h2og.weight) & (lstm_1997_layer.fc_h2og.weight <= init_upper))

  assert hasattr(lstm_1997_layer, 'fc_h2mc_in')
  assert isinstance(lstm_1997_layer.fc_h2mc_in, nn.Linear)
  assert lstm_1997_layer.fc_h2mc_in.weight.size() == torch.Size([n_blk * d_blk, n_blk * d_blk])
  assert lstm_1997_layer.fc_h2mc_in.bias is None
  assert torch.all(
    (init_lower <= lstm_1997_layer.fc_h2mc_in.weight) & (lstm_1997_layer.fc_h2mc_in.weight <= init_upper)
  )

  assert hasattr(lstm_1997_layer, 'c_0')
  assert isinstance(lstm_1997_layer.c_0, torch.Tensor)
  assert lstm_1997_layer.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all(lstm_1997_layer.c_0 == 0.0)

  assert hasattr(lstm_1997_layer, 'h_0')
  assert isinstance(lstm_1997_layer.h_0, torch.Tensor)
  assert lstm_1997_layer.h_0.size() == torch.Size([1, n_blk * d_blk])
  assert torch.all(lstm_1997_layer.h_0 == 0.0)


def test_lstm_1997_default_value(tknzr: BaseTknzr) -> None:
  """Ensure default value consistency."""
  d_emb = 1
  d_blk = 1
  init_ib = -1.0
  init_lower = -0.1
  init_ob = -1.0
  init_upper = 0.1
  label_smoothing = 0.0
  n_blk = 1
  n_lyr = 1
  p_emb = 0.0
  p_hid = 0.0
  lstm_1997 = LSTM1997(tknzr=tknzr)
  lstm_1997.params_init()

  assert hasattr(lstm_1997, 'd_blk')
  assert lstm_1997.d_blk == d_blk

  assert hasattr(lstm_1997, 'd_emb')
  assert lstm_1997.d_emb == d_emb

  assert hasattr(lstm_1997, 'd_hid')
  assert lstm_1997.d_hid == n_blk * d_blk

  assert hasattr(lstm_1997, 'init_ib')
  assert math.isclose(lstm_1997.init_ib, init_ib)

  assert hasattr(lstm_1997, 'init_lower')
  assert math.isclose(lstm_1997.init_lower, init_lower)

  assert hasattr(lstm_1997, 'init_ob')
  assert math.isclose(lstm_1997.init_ob, init_ob)

  assert hasattr(lstm_1997, 'init_upper')
  assert math.isclose(lstm_1997.init_upper, init_upper)

  assert hasattr(lstm_1997, 'label_smoothing')
  assert math.isclose(lstm_1997.label_smoothing, label_smoothing)

  assert hasattr(lstm_1997, 'n_blk')
  assert lstm_1997.n_blk == n_blk

  assert hasattr(lstm_1997, 'emb')
  assert isinstance(lstm_1997.emb, nn.Embedding)
  assert lstm_1997.emb.embedding_dim == d_emb
  assert lstm_1997.emb.num_embeddings == tknzr.vocab_size
  assert lstm_1997.emb.padding_idx == PAD_TKID
  assert torch.all((init_lower <= lstm_1997.emb.weight) & (lstm_1997.emb.weight <= init_upper))

  assert hasattr(lstm_1997, 'fc_e2h')
  assert isinstance(lstm_1997.fc_e2h, nn.Sequential)
  assert len(lstm_1997.fc_e2h) == 4
  assert isinstance(lstm_1997.fc_e2h[0], nn.Dropout)
  assert math.isclose(lstm_1997.fc_e2h[0].p, p_emb)
  assert isinstance(lstm_1997.fc_e2h[1], nn.Linear)
  assert lstm_1997.fc_e2h[1].weight.size() == torch.Size([n_blk * d_blk, d_emb])
  assert lstm_1997.fc_e2h[1].bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all((init_lower <= lstm_1997.fc_e2h[1].weight) & (lstm_1997.fc_e2h[1].weight <= init_upper))
  assert torch.all((init_lower <= lstm_1997.fc_e2h[1].bias) & (lstm_1997.fc_e2h[1].bias <= init_upper))
  assert isinstance(lstm_1997.fc_e2h[2], nn.Tanh)
  assert isinstance(lstm_1997.fc_e2h[3], nn.Dropout)
  assert math.isclose(lstm_1997.fc_e2h[3].p, p_hid)

  assert hasattr(lstm_1997, 'stack_rnn')
  assert isinstance(lstm_1997.stack_rnn, nn.ModuleList)
  assert len(lstm_1997.stack_rnn) == 2 * n_lyr
  for lyr in range(n_lyr):
    rnn_lyr = lstm_1997.stack_rnn[2 * lyr]
    assert isinstance(rnn_lyr, LSTM1997Layer)
    assert rnn_lyr.d_blk == d_blk
    assert rnn_lyr.in_feat == n_blk * d_blk
    assert rnn_lyr.n_blk == n_blk

    dropout_lyr = lstm_1997.stack_rnn[2 * lyr + 1]
    assert isinstance(dropout_lyr, nn.Dropout)
    assert math.isclose(dropout_lyr.p, p_hid)

  assert hasattr(lstm_1997, 'fc_h2e')
  assert isinstance(lstm_1997.fc_h2e, nn.Sequential)
  assert len(lstm_1997.fc_h2e) == 3
  assert isinstance(lstm_1997.fc_h2e[0], nn.Linear)
  assert lstm_1997.fc_h2e[0].weight.size() == torch.Size([d_emb, n_blk * d_blk])
  assert lstm_1997.fc_h2e[0].bias.size() == torch.Size([d_emb])
  assert torch.all((init_lower <= lstm_1997.fc_h2e[0].weight) & (lstm_1997.fc_h2e[0].weight <= init_upper))
  assert torch.all((init_lower <= lstm_1997.fc_h2e[0].bias) & (lstm_1997.fc_h2e[0].bias <= init_upper))
  assert isinstance(lstm_1997.fc_h2e[1], nn.Tanh)
  assert isinstance(lstm_1997.fc_h2e[2], nn.Dropout)
  assert math.isclose(lstm_1997.fc_h2e[2].p, p_hid)


def test_lstm_1997_parameters(
  d_blk: int,
  d_emb: int,
  init_ib: float,
  init_lower: float,
  init_ob: float,
  init_upper: float,
  label_smoothing: float,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
  lstm_1997: LSTM1997,
) -> None:
  """Must correctly construct parameters."""
  lstm_1997.params_init()

  assert hasattr(lstm_1997, 'd_blk')
  assert lstm_1997.d_blk == d_blk

  assert hasattr(lstm_1997, 'd_emb')
  assert lstm_1997.d_emb == d_emb

  assert hasattr(lstm_1997, 'd_hid')
  assert lstm_1997.d_hid == n_blk * d_blk

  assert hasattr(lstm_1997, 'init_ib')
  assert math.isclose(lstm_1997.init_ib, init_ib)

  assert hasattr(lstm_1997, 'init_lower')
  assert math.isclose(lstm_1997.init_lower, init_lower)

  assert hasattr(lstm_1997, 'init_ob')
  assert math.isclose(lstm_1997.init_ob, init_ob)

  assert hasattr(lstm_1997, 'init_upper')
  assert math.isclose(lstm_1997.init_upper, init_upper)

  assert hasattr(lstm_1997, 'label_smoothing')
  assert math.isclose(lstm_1997.label_smoothing, label_smoothing)

  assert hasattr(lstm_1997, 'n_blk')
  assert lstm_1997.n_blk == n_blk

  assert hasattr(lstm_1997, 'emb')
  assert isinstance(lstm_1997.emb, nn.Embedding)
  assert lstm_1997.emb.embedding_dim == d_emb
  assert lstm_1997.emb.num_embeddings == tknzr.vocab_size
  assert lstm_1997.emb.padding_idx == PAD_TKID
  assert torch.all((init_lower <= lstm_1997.emb.weight) & (lstm_1997.emb.weight <= init_upper))

  assert hasattr(lstm_1997, 'fc_e2h')
  assert isinstance(lstm_1997.fc_e2h, nn.Sequential)
  assert len(lstm_1997.fc_e2h) == 4
  assert isinstance(lstm_1997.fc_e2h[0], nn.Dropout)
  assert math.isclose(lstm_1997.fc_e2h[0].p, p_emb)
  assert isinstance(lstm_1997.fc_e2h[1], nn.Linear)
  assert lstm_1997.fc_e2h[1].weight.size() == torch.Size([n_blk * d_blk, d_emb])
  assert lstm_1997.fc_e2h[1].bias.size() == torch.Size([n_blk * d_blk])
  assert torch.all((init_lower <= lstm_1997.fc_e2h[1].weight) & (lstm_1997.fc_e2h[1].weight <= init_upper))
  assert torch.all((init_lower <= lstm_1997.fc_e2h[1].bias) & (lstm_1997.fc_e2h[1].bias <= init_upper))
  assert isinstance(lstm_1997.fc_e2h[2], nn.Tanh)
  assert isinstance(lstm_1997.fc_e2h[3], nn.Dropout)
  assert math.isclose(lstm_1997.fc_e2h[3].p, p_hid)

  assert hasattr(lstm_1997, 'stack_rnn')
  assert isinstance(lstm_1997.stack_rnn, nn.ModuleList)
  assert len(lstm_1997.stack_rnn) == 2 * n_lyr
  for lyr in range(n_lyr):
    rnn_lyr = lstm_1997.stack_rnn[2 * lyr]
    assert isinstance(rnn_lyr, LSTM1997Layer)
    assert rnn_lyr.d_blk == d_blk
    assert rnn_lyr.in_feat == n_blk * d_blk
    assert rnn_lyr.n_blk == n_blk

    dropout_lyr = lstm_1997.stack_rnn[2 * lyr + 1]
    assert isinstance(dropout_lyr, nn.Dropout)
    assert math.isclose(dropout_lyr.p, p_hid)

  assert hasattr(lstm_1997, 'fc_h2e')
  assert isinstance(lstm_1997.fc_h2e, nn.Sequential)
  assert len(lstm_1997.fc_h2e) == 3
  assert isinstance(lstm_1997.fc_h2e[0], nn.Linear)
  assert lstm_1997.fc_h2e[0].weight.size() == torch.Size([d_emb, n_blk * d_blk])
  assert lstm_1997.fc_h2e[0].bias.size() == torch.Size([d_emb])
  assert torch.all((init_lower <= lstm_1997.fc_h2e[0].weight) & (lstm_1997.fc_h2e[0].weight <= init_upper))
  assert torch.all((init_lower <= lstm_1997.fc_h2e[0].bias) & (lstm_1997.fc_h2e[0].bias <= init_upper))
  assert isinstance(lstm_1997.fc_h2e[1], nn.Tanh)
  assert isinstance(lstm_1997.fc_h2e[2], nn.Dropout)
  assert math.isclose(lstm_1997.fc_h2e[2].p, p_hid)
