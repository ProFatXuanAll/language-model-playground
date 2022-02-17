"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_2002.LSTM2002.forward`.
"""

import torch

from lmp.model._lstm_2002 import LSTM2002


def test_forward_path(
  lstm_2002: LSTM2002,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_2002 = lstm_2002.train()
  lstm_2002.zero_grad()

  loss = lstm_2002(batch_cur_tkids=batch_cur_tkids, batch_next_tkids=batch_next_tkids)
  loss.backward()

  assert loss.size() == torch.Size([])
  assert loss.dtype == torch.float
  assert hasattr(lstm_2002.emb.weight, 'grad')
  assert hasattr(lstm_2002.h_0, 'grad')
  assert hasattr(lstm_2002.c_0, 'grad')
  assert hasattr(lstm_2002.proj_e2cg[1].weight, 'grad')
  assert hasattr(lstm_2002.proj_e2cg[1].bias, 'grad')
  assert hasattr(lstm_2002.proj_h2cg.weight, 'grad')
  assert hasattr(lstm_2002.proj_c2ig, 'grad')
  assert hasattr(lstm_2002.proj_c2fg, 'grad')
  assert hasattr(lstm_2002.proj_c2og, 'grad')
  assert hasattr(lstm_2002.proj_h2e[1].weight, 'grad')
  assert hasattr(lstm_2002.proj_h2e[1].bias, 'grad')
