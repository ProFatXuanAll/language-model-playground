"""Test model construction.

Test target:
- :py:meth:`lmp.model._lstm_2000.LSTM2000.__init__`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._lstm_2000 import LSTM2000
from lmp.tknzr._base import PAD_TKID, BaseTknzr


def test_parameters(d_blk: int, d_emb: int, n_blk: int, tknzr: BaseTknzr, lstm_2000: LSTM2000) -> None:
  """Must correctly construct parameters."""
  d_hid = n_blk * d_blk
  inv_sqrt_dim = 1 / math.sqrt(max(d_emb, d_hid))

  assert hasattr(lstm_2000, 'emb')
  assert isinstance(lstm_2000.emb, nn.Embedding)
  assert lstm_2000.emb.embedding_dim == d_emb
  assert lstm_2000.emb.num_embeddings == tknzr.vocab_size
  assert lstm_2000.emb.padding_idx == PAD_TKID
  assert torch.all((-inv_sqrt_dim <= lstm_2000.emb.weight) & (lstm_2000.emb.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'proj_e2c')
  assert isinstance(lstm_2000.proj_e2c, nn.Linear)
  assert lstm_2000.proj_e2c.weight.size() == torch.Size([n_blk * (3 + d_blk), d_emb])
  assert lstm_2000.proj_e2c.bias.size() == torch.Size([n_blk * (3 + d_blk)])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.proj_e2c.weight) & (lstm_2000.proj_e2c.weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2000.proj_e2c.bias) & (lstm_2000.proj_e2c.bias <= inv_sqrt_dim))
  assert torch.all(
    (-inv_sqrt_dim <= lstm_2000.proj_e2c.bias[d_hid:d_hid + n_blk]) &
    (lstm_2000.proj_e2c.bias[d_hid:d_hid + n_blk] <= 0)
  )
  assert torch.all(
    (0 <= lstm_2000.proj_e2c.bias[d_hid + n_blk:d_hid + 2 * n_blk]) &
    (lstm_2000.proj_e2c.bias[d_hid + n_blk:d_hid + 2 * n_blk] <= inv_sqrt_dim)
  )
  assert torch.all(
    (-inv_sqrt_dim <= lstm_2000.proj_e2c.bias[d_hid + 2 * n_blk:]) & (lstm_2000.proj_e2c.bias[d_hid + 2 * n_blk:] <= 0)
  )

  assert hasattr(lstm_2000, 'proj_h2c')
  assert isinstance(lstm_2000.proj_h2c, nn.Linear)
  assert lstm_2000.proj_h2c.weight.size() == torch.Size([n_blk * (3 + d_blk), n_blk * d_blk])
  assert lstm_2000.proj_h2c.bias is None
  assert torch.all((-inv_sqrt_dim <= lstm_2000.proj_h2c.weight) & (lstm_2000.proj_h2c.weight <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'h_0')
  assert isinstance(lstm_2000.h_0, nn.Parameter)
  assert lstm_2000.h_0.size() == torch.Size([1, n_blk * d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.h_0) & (lstm_2000.h_0 <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'c_0')
  assert isinstance(lstm_2000.c_0, nn.Parameter)
  assert lstm_2000.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.c_0) & (lstm_2000.c_0 <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'proj_h2e')
  assert isinstance(lstm_2000.proj_h2e, nn.Linear)
  assert lstm_2000.proj_h2e.weight.size() == torch.Size([d_emb, n_blk * d_blk])
  assert lstm_2000.proj_h2e.bias.size() == torch.Size([d_emb])
  assert torch.all((-inv_sqrt_dim <= lstm_2000.proj_h2e.weight) & (lstm_2000.proj_h2e.weight <= inv_sqrt_dim))
  assert torch.all((-inv_sqrt_dim <= lstm_2000.proj_h2e.bias) & (lstm_2000.proj_h2e.bias <= inv_sqrt_dim))

  assert hasattr(lstm_2000, 'loss_fn')
  assert isinstance(lstm_2000.loss_fn, nn.CrossEntropyLoss)
  assert lstm_2000.loss_fn.ignore_index == tknzr.pad_tkid
