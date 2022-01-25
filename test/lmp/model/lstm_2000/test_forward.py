"""Test forward pass of :py:class:`lmp.model.LSTM2000`.

Test target:
- :py:meth:`lmp.model.LSTM2000.forward`.
"""

import torch

from lmp.model import LSTM2000


def test_forward_path(
  lstm_2000: LSTM2000,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_2000 = lstm_2000.train()
  lstm_2000.zero_grad()

  loss = lstm_2000(batch_cur_tkids=batch_cur_tkids, batch_next_tkids=batch_next_tkids)
  loss.backward()

  assert loss.size() == torch.Size([])
  assert loss.dtype == torch.float
  assert hasattr(lstm_2000.emb.weight, 'grad')
  assert hasattr(lstm_2000.h_0, 'grad')
  assert hasattr(lstm_2000.c_0, 'grad')
  assert hasattr(lstm_2000.proj_e2c.weight, 'grad')
  assert hasattr(lstm_2000.proj_e2c.bias, 'grad')
  assert hasattr(lstm_2000.proj_h2c.weight, 'grad')
  assert hasattr(lstm_2000.proj_h2e.weight, 'grad')
  assert hasattr(lstm_2000.proj_h2e.bias, 'grad')
