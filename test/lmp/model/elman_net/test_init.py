"""Test the construction of :py:class:`lmp.model.ElmanNet`.

Test target:
- :py:meth:`lmp.model.ElmanNet.__init__`.
"""

import torch
import torch.nn as nn

from lmp.model import ElmanNet
from lmp.tknzr import BaseTknzr


def test_parameters(d_emb: int, tknzr: BaseTknzr, elman_net: ElmanNet) -> None:
  """Must correctly construct parameters."""
  assert hasattr(elman_net, 'emb')
  assert isinstance(elman_net.emb, nn.Embedding)
  assert elman_net.emb.embedding_dim == d_emb
  assert elman_net.emb.num_embeddings == tknzr.vocab_size
  assert elman_net.emb.padding_idx == tknzr.pad_tkid

  assert hasattr(elman_net, 'proj_e2h')
  assert isinstance(elman_net.proj_e2h, nn.Linear)
  assert elman_net.proj_e2h.weight.size() == torch.Size([d_emb, d_emb])
  assert elman_net.proj_e2h.bias.size() == torch.Size([d_emb])

  assert hasattr(elman_net, 'proj_h2h')
  assert isinstance(elman_net.proj_h2h, nn.Linear)
  assert elman_net.proj_h2h.weight.size() == torch.Size([d_emb, d_emb])
  assert elman_net.proj_h2h.bias is None

  assert hasattr(elman_net, 'h_0')
  assert isinstance(elman_net.h_0, nn.Parameter)
  assert elman_net.h_0.size() == torch.Size([1, d_emb])

  assert hasattr(elman_net, 'loss_fn')
  assert isinstance(elman_net.loss_fn, nn.CrossEntropyLoss)
  assert elman_net.loss_fn.ignore_index == tknzr.pad_tkid
