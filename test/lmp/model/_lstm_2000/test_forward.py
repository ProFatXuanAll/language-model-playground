"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_2000.LSTM2000.forward`.
"""

import torch

from lmp.model._lstm_2000 import LSTM2000


def test_forward_path(
  lstm_2000: LSTM2000,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_2000 = lstm_2000.train()
  lstm_2000.zero_grad()

  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    loss, batch_prev_states = lstm_2000.loss(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_next_tkids=batch_next_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )
    loss.backward()

    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float
    assert hasattr(lstm_2000.c_0, 'grad')
    assert hasattr(lstm_2000.emb.weight, 'grad')
    assert hasattr(lstm_2000.fc_e2fg[1].weight, 'grad')
    assert hasattr(lstm_2000.fc_e2fg[1].bias, 'grad')
    assert hasattr(lstm_2000.fc_e2ig[1].weight, 'grad')
    assert hasattr(lstm_2000.fc_e2ig[1].bias, 'grad')
    assert hasattr(lstm_2000.fc_e2mc_in[1].weight, 'grad')
    assert hasattr(lstm_2000.fc_e2mc_in[1].bias, 'grad')
    assert hasattr(lstm_2000.fc_e2og[1].weight, 'grad')
    assert hasattr(lstm_2000.fc_e2og[1].bias, 'grad')
    assert hasattr(lstm_2000.fc_h2e[1].weight, 'grad')
    assert hasattr(lstm_2000.fc_h2e[1].bias, 'grad')
    assert hasattr(lstm_2000.fc_h2fg.weight, 'grad')
    assert hasattr(lstm_2000.fc_h2ig.weight, 'grad')
    assert hasattr(lstm_2000.fc_h2mc_in.weight, 'grad')
    assert hasattr(lstm_2000.fc_h2og.weight, 'grad')
    assert hasattr(lstm_2000.h_0, 'grad')
