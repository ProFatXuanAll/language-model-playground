"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.forward`.
"""

import torch

from lmp.model._lstm_1997 import LSTM1997


def test_forward_path(
  lstm_1997: LSTM1997,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_1997 = lstm_1997.train()
  lstm_1997.zero_grad()

  loss = lstm_1997(batch_cur_tkids=batch_cur_tkids, batch_next_tkids=batch_next_tkids)
  loss.backward()

  assert loss.size() == torch.Size([])
  assert loss.dtype == torch.float
  assert hasattr(lstm_1997.emb.weight, 'grad')
  assert hasattr(lstm_1997.h_0, 'grad')
  assert hasattr(lstm_1997.c_0, 'grad')
  assert hasattr(lstm_1997.proj_e2cg[1].weight, 'grad')
  assert hasattr(lstm_1997.proj_e2cg[1].bias, 'grad')
  assert hasattr(lstm_1997.proj_h2cg.weight, 'grad')
  assert hasattr(lstm_1997.proj_h2e[1].weight, 'grad')
  assert hasattr(lstm_1997.proj_h2e[1].bias, 'grad')
