"""Test model construction.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.__init__`.
- :py:meth:`lmp.model._elman_net.ElmanNetLayer.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._elman_net import ElmanNet, ElmanNetLayer
from lmp.tknzr._base import PAD_TKID, BaseTknzr


def test_elman_net_layer_parameters(
  n_feat: int,
  elman_net_layer: ElmanNetLayer,
) -> None:
  """Must correctly construct parameters."""
  elman_net_layer.params_init()
  inv_sqrt_dim = 1 / math.sqrt(n_feat)

  assert elman_net_layer.n_feat == n_feat

  assert hasattr(elman_net_layer, 'fc_x2h')
  assert isinstance(elman_net_layer.fc_x2h, nn.Linear)
  assert elman_net_layer.fc_x2h.weight.size() == torch.Size([n_feat, n_feat])
  assert elman_net_layer.fc_x2h.bias.size() == torch.Size([n_feat])
  assert torch.all((-inv_sqrt_dim <= elman_net_layer.fc_x2h.weight) & (elman_net_layer.fc_x2h.weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= elman_net_layer.fc_x2h.bias) & (elman_net_layer.fc_x2h.bias <= inv_sqrt_dim))

  assert hasattr(elman_net_layer, 'fc_h2h')
  assert isinstance(elman_net_layer.fc_h2h, nn.Linear)
  assert elman_net_layer.fc_h2h.weight.size() == torch.Size([n_feat, n_feat])
  assert elman_net_layer.fc_h2h.bias is None
  assert torch.all((-inv_sqrt_dim <= elman_net_layer.fc_h2h.weight) & (elman_net_layer.fc_h2h.weight <= inv_sqrt_dim))

  assert hasattr(elman_net_layer, 'h_0')
  assert isinstance(elman_net_layer.h_0, nn.Parameter)
  assert elman_net_layer.h_0.size() == torch.Size([1, n_feat])
  assert torch.all((-inv_sqrt_dim <= elman_net_layer.h_0) & (elman_net_layer.h_0 <= inv_sqrt_dim))


def test_elman_net_parameters(
  d_emb: int,
  d_hid: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
  elman_net: ElmanNet,
) -> None:
  """Must correctly construct parameters."""
  elman_net.params_init()
  inv_sqrt_dim = 1 / math.sqrt(max(d_emb, d_hid))

  assert elman_net.d_emb == d_emb
  assert elman_net.d_hid == d_hid
  assert elman_net.n_lyr == n_lyr
  assert math.isclose(elman_net.p_emb, p_emb)
  assert math.isclose(elman_net.p_hid, p_hid)

  assert hasattr(elman_net, 'emb')
  assert isinstance(elman_net.emb, nn.Embedding)
  assert elman_net.emb.embedding_dim == d_emb
  assert elman_net.emb.num_embeddings == tknzr.vocab_size
  assert elman_net.emb.padding_idx == PAD_TKID
  assert torch.all((-inv_sqrt_dim <= elman_net.emb.weight) & (elman_net.emb.weight <= inv_sqrt_dim))

  assert hasattr(elman_net, 'fc_e2h')
  assert isinstance(elman_net.fc_e2h, nn.Sequential)
  assert len(elman_net.fc_e2h) == 4
  assert isinstance(elman_net.fc_e2h[0], nn.Dropout)
  assert math.isclose(elman_net.fc_e2h[0].p, p_emb)
  assert isinstance(elman_net.fc_e2h[1], nn.Linear)
  assert elman_net.fc_e2h[1].weight.size() == torch.Size([d_hid, d_emb])
  assert elman_net.fc_e2h[1].bias.size() == torch.Size([d_hid])
  assert torch.all((-inv_sqrt_dim <= elman_net.fc_e2h[1].weight) & (elman_net.fc_e2h[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= elman_net.fc_e2h[1].bias) & (elman_net.fc_e2h[1].bias <= inv_sqrt_dim))
  assert isinstance(elman_net.fc_e2h[2], nn.Tanh)
  assert isinstance(elman_net.fc_e2h[3], nn.Dropout)
  assert math.isclose(elman_net.fc_e2h[3].p, p_hid)

  assert hasattr(elman_net, 'stack_rnn')
  assert isinstance(elman_net.stack_rnn, nn.ModuleList)
  assert len(elman_net.stack_rnn) == 2 * n_lyr
  for lyr in range(n_lyr):
    rnn_lyr = elman_net.stack_rnn[2 * lyr]
    assert isinstance(rnn_lyr, ElmanNetLayer)
    assert rnn_lyr.n_feat == d_hid

    dropout_lyr = elman_net.stack_rnn[2 * lyr + 1]
    assert isinstance(dropout_lyr, nn.Dropout)
    assert math.isclose(dropout_lyr.p, p_hid)

  assert hasattr(elman_net, 'fc_h2e')
  assert isinstance(elman_net.fc_h2e, nn.Sequential)
  assert len(elman_net.fc_h2e) == 3
  assert isinstance(elman_net.fc_h2e[0], nn.Linear)
  assert elman_net.fc_h2e[0].weight.size() == torch.Size([d_emb, d_hid])
  assert elman_net.fc_h2e[0].bias.size() == torch.Size([d_emb])
  assert torch.all((-inv_sqrt_dim <= elman_net.fc_h2e[0].weight) & (elman_net.fc_h2e[0].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= elman_net.fc_h2e[0].bias) & (elman_net.fc_h2e[0].bias <= inv_sqrt_dim))
  assert isinstance(elman_net.fc_h2e[1], nn.Tanh)
  assert isinstance(elman_net.fc_h2e[2], nn.Dropout)
  assert math.isclose(elman_net.fc_h2e[2].p, p_hid)
