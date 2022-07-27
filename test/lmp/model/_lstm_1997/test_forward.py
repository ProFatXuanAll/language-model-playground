"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.forward`.
- :py:meth:`lmp.model._lstm_1997.LSTM1997Layer.forward`.
"""

import torch

from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer


def test_lstm_1997_layer_forward_path(
  lstm_1997_layer: LSTM1997Layer,
  batch_x: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_1997_layer = lstm_1997_layer.train()
  lstm_1997_layer.zero_grad()

  batch_prev_states = None
  for idx in range(batch_x.size(1)):
    batch_cur_states = lstm_1997_layer(
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
    assert hasattr(lstm_1997_layer.fc_x2ig.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2ig.bias, 'grad')
    assert hasattr(lstm_1997_layer.fc_h2ig.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2og.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2og.bias, 'grad')
    assert hasattr(lstm_1997_layer.fc_h2og.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2mc_in.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2mc_in.bias, 'grad')
    assert hasattr(lstm_1997_layer.fc_h2mc_in.weight, 'grad')

    if idx == 0:
      assert hasattr(lstm_1997_layer.h_0, 'grad')
      assert hasattr(lstm_1997_layer.c_0, 'grad')

    lstm_1997_layer.zero_grad()


def test_lstm_1997_forward_path(
  lstm_1997: LSTM1997,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_1997 = lstm_1997.train()
  lstm_1997.zero_grad()

  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    loss, batch_prev_states = lstm_1997.loss(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_next_tkids=batch_next_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )
    loss.backward()

    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float
    assert hasattr(lstm_1997.emb.weight, 'grad')
    assert hasattr(lstm_1997.fc_e2h[1].weight, 'grad')
    assert hasattr(lstm_1997.fc_e2h[1].bias, 'grad')
    for lyr in range(lstm_1997.n_lyr):
      assert hasattr(lstm_1997.stack_rnn[2 * lyr].fc_x2ig.weight, 'grad')
    assert hasattr(lstm_1997.fc_h2e[0].weight, 'grad')
    assert hasattr(lstm_1997.fc_h2e[0].bias, 'grad')

    lstm_1997.zero_grad()
