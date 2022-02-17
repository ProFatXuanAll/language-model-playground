"""Test model construction.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._elman_net import ElmanNet
from lmp.tknzr._base import PAD_TKID, BaseTknzr


def test_parameters(d_emb: int, d_hid: int, p_emb: float, p_hid: float, tknzr: BaseTknzr, elman_net: ElmanNet) -> None:
  """Must correctly construct parameters."""
  inv_sqrt_dim = 1 / math.sqrt(d_emb)

  assert hasattr(elman_net, 'emb')
  assert isinstance(elman_net.emb, nn.Embedding)
  assert elman_net.emb.embedding_dim == d_emb
  assert elman_net.emb.num_embeddings == tknzr.vocab_size
  assert elman_net.emb.padding_idx == PAD_TKID
  assert torch.all((-inv_sqrt_dim <= elman_net.emb.weight) & (elman_net.emb.weight <= inv_sqrt_dim))

  assert hasattr(elman_net, 'proj_e2h')
  assert isinstance(elman_net.proj_e2h, nn.Sequential)
  assert len(elman_net.proj_e2h) == 2
  assert isinstance(elman_net.proj_e2h[0], nn.Dropout)
  assert math.isclose(elman_net.proj_e2h[0].p, p_emb)
  assert isinstance(elman_net.proj_e2h[1], nn.Linear)
  assert elman_net.proj_e2h[1].weight.size() == torch.Size([d_hid, d_emb])
  assert elman_net.proj_e2h[1].bias.size() == torch.Size([d_hid])
  assert torch.all((-inv_sqrt_dim <= elman_net.proj_e2h[1].weight) & (elman_net.proj_e2h[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= elman_net.proj_e2h[1].bias) & (elman_net.proj_e2h[1].bias <= inv_sqrt_dim))

  assert hasattr(elman_net, 'proj_h2h')
  assert isinstance(elman_net.proj_h2h, nn.Linear)
  assert elman_net.proj_h2h.weight.size() == torch.Size([d_hid, d_hid])
  assert elman_net.proj_h2h.bias is None
  assert torch.all((-inv_sqrt_dim <= elman_net.proj_h2h.weight) & (elman_net.proj_h2h.weight <= inv_sqrt_dim))

  assert hasattr(elman_net, 'h_0')
  assert isinstance(elman_net.h_0, nn.Parameter)
  assert elman_net.h_0.size() == torch.Size([1, d_hid])
  assert torch.all((-inv_sqrt_dim <= elman_net.h_0) & (elman_net.h_0 <= inv_sqrt_dim))

  assert hasattr(elman_net, 'proj_h2e')
  assert isinstance(elman_net.proj_h2e, nn.Sequential)
  assert len(elman_net.proj_h2e) == 4
  assert isinstance(elman_net.proj_h2e[0], nn.Dropout)
  assert math.isclose(elman_net.proj_h2e[0].p, p_hid)
  assert isinstance(elman_net.proj_h2e[1], nn.Linear)
  assert elman_net.proj_h2e[1].weight.size() == torch.Size([d_emb, d_hid])
  assert elman_net.proj_h2e[1].bias.size() == torch.Size([d_emb])
  assert torch.all((-inv_sqrt_dim <= elman_net.proj_h2e[1].weight) & (elman_net.proj_h2e[1].weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= elman_net.proj_h2e[1].bias) & (elman_net.proj_h2e[1].bias <= inv_sqrt_dim))
  assert isinstance(elman_net.proj_h2e[2], nn.Tanh)
  assert isinstance(elman_net.proj_h2e[3], nn.Dropout)
  assert math.isclose(elman_net.proj_h2e[3].p, p_hid)
