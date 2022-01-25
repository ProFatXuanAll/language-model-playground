"""Test the construction of :py:class:`lmp.model.LSTM1997`.

Test target:
- :py:meth:`lmp.model.LSTM1997.__init__`.
"""

import torch
import torch.nn as nn

from lmp.model import LSTM1997
from lmp.tknzr import BaseTknzr


def test_parameters(d_cell: int, d_emb: int, n_cell: int, tknzr: BaseTknzr, lstm_1997: LSTM1997) -> None:
  """Must correctly construct parameters."""
  assert hasattr(lstm_1997, 'emb')
  assert isinstance(lstm_1997.emb, nn.Embedding)
  assert lstm_1997.emb.embedding_dim == d_emb
  assert lstm_1997.emb.num_embeddings == tknzr.vocab_size
  assert lstm_1997.emb.padding_idx == tknzr.pad_tkid

  assert hasattr(lstm_1997, 'proj_e2c')
  assert isinstance(lstm_1997.proj_e2c, nn.Linear)
  assert lstm_1997.proj_e2c.weight.size() == torch.Size([n_cell * (2 + d_cell), d_emb])
  assert lstm_1997.proj_e2c.bias.size() == torch.Size([n_cell * (2 + d_cell)])

  assert hasattr(lstm_1997, 'proj_h2c')
  assert isinstance(lstm_1997.proj_h2c, nn.Linear)
  assert lstm_1997.proj_h2c.weight.size() == torch.Size([n_cell * (2 + d_cell), n_cell * d_cell])
  assert lstm_1997.proj_h2c.bias is None

  assert hasattr(lstm_1997, 'h_0')
  assert isinstance(lstm_1997.h_0, nn.Parameter)
  assert lstm_1997.h_0.size() == torch.Size([1, n_cell * d_cell])

  assert hasattr(lstm_1997, 'c_0')
  assert isinstance(lstm_1997.c_0, nn.Parameter)
  assert lstm_1997.c_0.size() == torch.Size([1, n_cell, d_cell])

  assert hasattr(lstm_1997, 'proj_h2e')
  assert isinstance(lstm_1997.proj_h2e, nn.Linear)
  assert lstm_1997.proj_h2e.weight.size() == torch.Size([d_emb, n_cell * d_cell])
  assert lstm_1997.proj_h2e.bias.size() == torch.Size([d_emb])

  assert hasattr(lstm_1997, 'loss_fn')
  assert isinstance(lstm_1997.loss_fn, nn.CrossEntropyLoss)
  assert lstm_1997.loss_fn.ignore_index == tknzr.pad_tkid
