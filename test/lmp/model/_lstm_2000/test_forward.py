"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_2000.LSTM2000.forward`.
- :py:meth:`lmp.model._lstm_2000.LSTM2000Layer.forward`.
"""

import torch

from lmp.model._lstm_2000 import LSTM2000, LSTM2000Layer


def test_lstm_2000_layer_forward_path(
  lstm_2000_layer: LSTM2000Layer,
  batch_x: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_2000_layer = lstm_2000_layer.train()
  lstm_2000_layer.zero_grad()

  batch_prev_states = None
  for idx in range(batch_x.size(1)):
    batch_cur_states = lstm_2000_layer(
      batch_x=batch_x[:, idx, :].unsqueeze(1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(batch_cur_states, list)
    assert len(batch_cur_states) == 2

    batch_h_prev = batch_cur_states[0].detach()
    batch_c_prev = batch_cur_states[1].detach()
    batch_prev_states = [batch_h_prev[:, -1, :], batch_c_prev[:, -1, :]]

    loss = batch_cur_states[0].sum()
    loss.backward()

    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float
    assert hasattr(lstm_2000_layer.fc_x2fg.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2fg.bias, 'grad')
    assert hasattr(lstm_2000_layer.fc_h2fg.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2ig.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2ig.bias, 'grad')
    assert hasattr(lstm_2000_layer.fc_h2ig.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2og.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2og.bias, 'grad')
    assert hasattr(lstm_2000_layer.fc_h2og.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2mc_in.weight, 'grad')
    assert hasattr(lstm_2000_layer.fc_x2mc_in.bias, 'grad')
    assert hasattr(lstm_2000_layer.fc_h2mc_in.weight, 'grad')

    if idx == 0:
      assert hasattr(lstm_2000_layer.h_0, 'grad')
      assert hasattr(lstm_2000_layer.c_0, 'grad')

    lstm_2000_layer.zero_grad()


def test_lstm_2000_forward_path(
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
    assert hasattr(lstm_2000.emb.weight, 'grad')
    assert hasattr(lstm_2000.fc_e2h[1].weight, 'grad')
    assert hasattr(lstm_2000.fc_e2h[1].bias, 'grad')
    for lyr in range(lstm_2000.n_lyr):
      assert hasattr(lstm_2000.stack_rnn[2 * lyr].fc_x2ig.weight, 'grad')
    assert hasattr(lstm_2000.fc_h2e[0].weight, 'grad')
    assert hasattr(lstm_2000.fc_h2e[0].bias, 'grad')

    lstm_2000.zero_grad()
