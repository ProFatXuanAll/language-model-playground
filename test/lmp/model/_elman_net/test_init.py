"""Test model construction.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.__init__`.
- :py:meth:`lmp.model._elman_net.ElmanNet.params_init`.
- :py:meth:`lmp.model._elman_net.ElmanNetLayer.__init__`.
- :py:meth:`lmp.model._elman_net.ElmanNetLayer.params_init`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._elman_net import ElmanNet, ElmanNetLayer
from lmp.tknzr._base import BaseTknzr
from lmp.vars import PAD_TKID


def test_elman_net_layer_default_value() -> None:
  """Ensure default value consistency."""
  in_feat = 1
  init_lower = -0.1
  init_upper = 0.1
  out_feat = 1
  elman_net_layer = ElmanNetLayer()
  elman_net_layer.params_init()

  assert hasattr(elman_net_layer, 'in_feat')
  assert elman_net_layer.in_feat == in_feat

  assert hasattr(elman_net_layer, 'init_lower')
  assert math.isclose(elman_net_layer.init_lower, init_lower)

  assert hasattr(elman_net_layer, 'init_upper')
  assert math.isclose(elman_net_layer.init_upper, init_upper)

  assert hasattr(elman_net_layer, 'out_feat')
  assert elman_net_layer.out_feat == out_feat

  assert hasattr(elman_net_layer, 'fc_x2h')
  assert isinstance(elman_net_layer.fc_x2h, nn.Linear)
  assert elman_net_layer.fc_x2h.weight.size() == torch.Size([out_feat, in_feat])
  assert elman_net_layer.fc_x2h.bias.size() == torch.Size([out_feat])
  assert torch.all((init_lower <= elman_net_layer.fc_x2h.weight) & (elman_net_layer.fc_x2h.weight <= init_upper))
  assert torch.all((init_lower <= elman_net_layer.fc_x2h.bias) & (elman_net_layer.fc_x2h.bias <= init_upper))

  assert hasattr(elman_net_layer, 'fc_h2h')
  assert isinstance(elman_net_layer.fc_h2h, nn.Linear)
  assert elman_net_layer.fc_h2h.weight.size() == torch.Size([out_feat, out_feat])
  assert elman_net_layer.fc_h2h.bias is None
  assert torch.all((init_lower <= elman_net_layer.fc_h2h.weight) & (elman_net_layer.fc_h2h.weight <= init_upper))

  assert hasattr(elman_net_layer, 'h_0')
  assert isinstance(elman_net_layer.h_0, torch.Tensor)
  assert elman_net_layer.h_0.size() == torch.Size([1, out_feat])
  assert torch.all((init_lower <= elman_net_layer.h_0) & (elman_net_layer.h_0 <= init_upper))


def test_elman_net_layer_parameters(
  elman_net_layer: ElmanNetLayer,
  in_feat: int,
  init_lower: float,
  init_upper: float,
  out_feat: int,
) -> None:
  """Must correctly construct parameters."""
  elman_net_layer.params_init()

  assert hasattr(elman_net_layer, 'in_feat')
  assert elman_net_layer.in_feat == in_feat

  assert hasattr(elman_net_layer, 'init_lower')
  assert math.isclose(elman_net_layer.init_lower, init_lower)

  assert hasattr(elman_net_layer, 'init_upper')
  assert math.isclose(elman_net_layer.init_upper, init_upper)

  assert hasattr(elman_net_layer, 'out_feat')
  assert elman_net_layer.out_feat == out_feat

  assert hasattr(elman_net_layer, 'fc_x2h')
  assert isinstance(elman_net_layer.fc_x2h, nn.Linear)
  assert elman_net_layer.fc_x2h.weight.size() == torch.Size([out_feat, in_feat])
  assert elman_net_layer.fc_x2h.bias.size() == torch.Size([out_feat])
  assert torch.all((init_lower <= elman_net_layer.fc_x2h.weight) & (elman_net_layer.fc_x2h.weight <= init_upper))
  assert torch.all((init_lower <= elman_net_layer.fc_x2h.bias) & (elman_net_layer.fc_x2h.bias <= init_upper))

  assert hasattr(elman_net_layer, 'fc_h2h')
  assert isinstance(elman_net_layer.fc_h2h, nn.Linear)
  assert elman_net_layer.fc_h2h.weight.size() == torch.Size([out_feat, out_feat])
  assert elman_net_layer.fc_h2h.bias is None
  assert torch.all((init_lower <= elman_net_layer.fc_h2h.weight) & (elman_net_layer.fc_h2h.weight <= init_upper))

  assert hasattr(elman_net_layer, 'h_0')
  assert isinstance(elman_net_layer.h_0, torch.Tensor)
  assert elman_net_layer.h_0.size() == torch.Size([1, out_feat])
  assert torch.all(elman_net_layer.h_0 == 0.0)


def test_elman_net_default_value(tknzr: BaseTknzr) -> None:
  """Ensure default value consistency."""
  d_emb = 1
  d_hid = 1
  init_lower = -0.1
  init_upper = 0.1
  label_smoothing = 0.0
  n_lyr = 1
  p_emb = 0.0
  p_hid = 0.0
  elman_net = ElmanNet(tknzr=tknzr)
  elman_net.params_init()

  assert hasattr(elman_net, 'd_emb')
  assert elman_net.d_emb == d_emb

  assert hasattr(elman_net, 'd_hid')
  assert elman_net.d_hid == d_hid

  assert hasattr(elman_net, 'init_lower')
  assert math.isclose(elman_net.init_lower, init_lower)

  assert hasattr(elman_net, 'init_upper')
  assert math.isclose(elman_net.init_upper, init_upper)

  assert hasattr(elman_net, 'label_smoothing')
  assert math.isclose(elman_net.label_smoothing, label_smoothing)

  assert hasattr(elman_net, 'n_lyr')
  assert elman_net.n_lyr == n_lyr

  assert hasattr(elman_net, 'p_emb')
  assert math.isclose(elman_net.p_emb, p_emb)

  assert hasattr(elman_net, 'p_hid')
  assert math.isclose(elman_net.p_hid, p_hid)

  assert hasattr(elman_net, 'emb')
  assert isinstance(elman_net.emb, nn.Embedding)
  assert elman_net.emb.embedding_dim == d_emb
  assert elman_net.emb.num_embeddings == tknzr.vocab_size
  assert elman_net.emb.padding_idx == PAD_TKID
  assert torch.all((init_lower <= elman_net.emb.weight) & (elman_net.emb.weight <= init_upper))

  assert hasattr(elman_net, 'fc_e2h')
  assert isinstance(elman_net.fc_e2h, nn.Sequential)
  assert len(elman_net.fc_e2h) == 4
  assert isinstance(elman_net.fc_e2h[0], nn.Dropout)
  assert math.isclose(elman_net.fc_e2h[0].p, p_emb)
  assert isinstance(elman_net.fc_e2h[1], nn.Linear)
  assert elman_net.fc_e2h[1].weight.size() == torch.Size([d_hid, d_emb])
  assert elman_net.fc_e2h[1].bias.size() == torch.Size([d_hid])
  assert torch.all((init_lower <= elman_net.fc_e2h[1].weight) & (elman_net.fc_e2h[1].weight <= init_upper))
  assert torch.all((init_lower <= elman_net.fc_e2h[1].bias) & (elman_net.fc_e2h[1].bias <= init_upper))
  assert isinstance(elman_net.fc_e2h[2], nn.Tanh)
  assert isinstance(elman_net.fc_e2h[3], nn.Dropout)
  assert math.isclose(elman_net.fc_e2h[3].p, p_hid)

  assert hasattr(elman_net, 'stack_rnn')
  assert isinstance(elman_net.stack_rnn, nn.ModuleList)
  assert len(elman_net.stack_rnn) == 2 * n_lyr
  for lyr in range(n_lyr):
    rnn_lyr = elman_net.stack_rnn[2 * lyr]
    assert isinstance(rnn_lyr, ElmanNetLayer)
    assert rnn_lyr.in_feat == d_hid
    assert rnn_lyr.out_feat == d_hid

    dropout_lyr = elman_net.stack_rnn[2 * lyr + 1]
    assert isinstance(dropout_lyr, nn.Dropout)
    assert math.isclose(dropout_lyr.p, p_hid)

  assert hasattr(elman_net, 'fc_h2e')
  assert isinstance(elman_net.fc_h2e, nn.Sequential)
  assert len(elman_net.fc_h2e) == 3
  assert isinstance(elman_net.fc_h2e[0], nn.Linear)
  assert elman_net.fc_h2e[0].weight.size() == torch.Size([d_emb, d_hid])
  assert elman_net.fc_h2e[0].bias.size() == torch.Size([d_emb])
  assert torch.all((init_lower <= elman_net.fc_h2e[0].weight) & (elman_net.fc_h2e[0].weight <= init_upper))
  assert torch.all((init_lower <= elman_net.fc_h2e[0].bias) & (elman_net.fc_h2e[0].bias <= init_upper))
  assert isinstance(elman_net.fc_h2e[1], nn.Tanh)
  assert isinstance(elman_net.fc_h2e[2], nn.Dropout)
  assert math.isclose(elman_net.fc_h2e[2].p, p_hid)

  assert hasattr(elman_net, 'loss_fn')
  assert isinstance(elman_net.loss_fn, nn.CrossEntropyLoss)
  assert elman_net.loss_fn.ignore_index == PAD_TKID
  assert math.isclose(elman_net.loss_fn.label_smoothing, label_smoothing)


def test_elman_net_parameters(
  d_emb: int,
  d_hid: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
  elman_net: ElmanNet,
) -> None:
  """Must correctly construct parameters."""
  elman_net.params_init()

  assert hasattr(elman_net, 'd_emb')
  assert elman_net.d_emb == d_emb

  assert hasattr(elman_net, 'd_hid')
  assert elman_net.d_hid == d_hid

  assert hasattr(elman_net, 'init_lower')
  assert math.isclose(elman_net.init_lower, init_lower)

  assert hasattr(elman_net, 'init_upper')
  assert math.isclose(elman_net.init_upper, init_upper)

  assert hasattr(elman_net, 'label_smoothing')
  assert math.isclose(elman_net.label_smoothing, label_smoothing)

  assert hasattr(elman_net, 'n_lyr')
  assert elman_net.n_lyr == n_lyr

  assert hasattr(elman_net, 'p_emb')
  assert math.isclose(elman_net.p_emb, p_emb)

  assert hasattr(elman_net, 'p_hid')
  assert math.isclose(elman_net.p_hid, p_hid)

  assert hasattr(elman_net, 'emb')
  assert isinstance(elman_net.emb, nn.Embedding)
  assert elman_net.emb.embedding_dim == d_emb
  assert elman_net.emb.num_embeddings == tknzr.vocab_size
  assert elman_net.emb.padding_idx == PAD_TKID
  assert torch.all((init_lower <= elman_net.emb.weight) & (elman_net.emb.weight <= init_upper))

  assert hasattr(elman_net, 'fc_e2h')
  assert isinstance(elman_net.fc_e2h, nn.Sequential)
  assert len(elman_net.fc_e2h) == 4
  assert isinstance(elman_net.fc_e2h[0], nn.Dropout)
  assert math.isclose(elman_net.fc_e2h[0].p, p_emb)
  assert isinstance(elman_net.fc_e2h[1], nn.Linear)
  assert elman_net.fc_e2h[1].weight.size() == torch.Size([d_hid, d_emb])
  assert elman_net.fc_e2h[1].bias.size() == torch.Size([d_hid])
  assert torch.all((init_lower <= elman_net.fc_e2h[1].weight) & (elman_net.fc_e2h[1].weight <= init_upper))
  assert torch.all((init_lower <= elman_net.fc_e2h[1].bias) & (elman_net.fc_e2h[1].bias <= init_upper))
  assert isinstance(elman_net.fc_e2h[2], nn.Tanh)
  assert isinstance(elman_net.fc_e2h[3], nn.Dropout)
  assert math.isclose(elman_net.fc_e2h[3].p, p_hid)

  assert hasattr(elman_net, 'stack_rnn')
  assert isinstance(elman_net.stack_rnn, nn.ModuleList)
  assert len(elman_net.stack_rnn) == 2 * n_lyr
  for lyr in range(n_lyr):
    rnn_lyr = elman_net.stack_rnn[2 * lyr]
    assert isinstance(rnn_lyr, ElmanNetLayer)
    assert rnn_lyr.in_feat == d_hid
    assert rnn_lyr.out_feat == d_hid

    dropout_lyr = elman_net.stack_rnn[2 * lyr + 1]
    assert isinstance(dropout_lyr, nn.Dropout)
    assert math.isclose(dropout_lyr.p, p_hid)

  assert hasattr(elman_net, 'fc_h2e')
  assert isinstance(elman_net.fc_h2e, nn.Sequential)
  assert len(elman_net.fc_h2e) == 3
  assert isinstance(elman_net.fc_h2e[0], nn.Linear)
  assert elman_net.fc_h2e[0].weight.size() == torch.Size([d_emb, d_hid])
  assert elman_net.fc_h2e[0].bias.size() == torch.Size([d_emb])
  assert torch.all((init_lower <= elman_net.fc_h2e[0].weight) & (elman_net.fc_h2e[0].weight <= init_upper))
  assert torch.all((init_lower <= elman_net.fc_h2e[0].bias) & (elman_net.fc_h2e[0].bias <= init_upper))
  assert isinstance(elman_net.fc_h2e[1], nn.Tanh)
  assert isinstance(elman_net.fc_h2e[2], nn.Dropout)
  assert math.isclose(elman_net.fc_h2e[2].p, p_hid)

  assert hasattr(elman_net, 'loss_fn')
  assert isinstance(elman_net.loss_fn, nn.CrossEntropyLoss)
  assert elman_net.loss_fn.ignore_index == PAD_TKID
  assert math.isclose(elman_net.loss_fn.label_smoothing, label_smoothing)
