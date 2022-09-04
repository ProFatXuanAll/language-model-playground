"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.forward`.
- :py:meth:`lmp.model._lstm_1997.LSTM1997Layer.forward`.
"""

import torch

from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer


def test_lstm_1997_layer_forward_path(lstm_1997_layer: LSTM1997Layer, x: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_1997_layer = lstm_1997_layer.train()
  lstm_1997_layer.zero_grad()

  B = x.size(0)
  c_0 = None
  h_0 = None
  for idx in range(x.size(1)):
    c, h = lstm_1997_layer(x=x[:, idx, :].unsqueeze(1), c_0=c_0, h_0=h_0)

    assert isinstance(c, torch.Tensor)
    assert c.size() == torch.Size([B, 1, lstm_1997_layer.n_blk, lstm_1997_layer.d_blk])
    assert c.dtype == torch.float
    assert isinstance(h, torch.Tensor)
    assert h.size() == torch.Size([B, 1, lstm_1997_layer.d_hid])
    assert h.dtype == torch.float

    logits = h.sum()
    logits.backward()

    assert hasattr(lstm_1997_layer.fc_x2ig.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2ig.bias, 'grad')
    assert hasattr(lstm_1997_layer.fc_h2ig.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2og.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2og.bias, 'grad')
    assert hasattr(lstm_1997_layer.fc_h2og.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2mc_in.weight, 'grad')
    assert hasattr(lstm_1997_layer.fc_x2mc_in.bias, 'grad')
    assert hasattr(lstm_1997_layer.fc_h2mc_in.weight, 'grad')

    lstm_1997_layer.zero_grad()
    c_0 = c.detach()[:, -1, :, :]
    h_0 = h.detach()[:, -1, :]


def test_lstm_1997_forward_path(lstm_1997: LSTM1997, batch_cur_tkids: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_1997 = lstm_1997.train()
  lstm_1997.zero_grad()

  B = batch_cur_tkids.size(0)
  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    logits, batch_cur_states = lstm_1997(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(logits, torch.Tensor)
    assert logits.size() == torch.Size([B, 1, lstm_1997.emb.num_embeddings])
    assert logits.dtype == torch.float

    logits.sum().backward()
    assert hasattr(lstm_1997.emb.weight, 'grad')
    assert hasattr(lstm_1997.fc_e2h[1].weight, 'grad')
    assert hasattr(lstm_1997.fc_e2h[1].bias, 'grad')

    for lyr in range(lstm_1997.n_lyr):
      assert hasattr(lstm_1997.stack_rnn[2 * lyr].fc_x2ig.weight, 'grad')

    assert hasattr(lstm_1997.fc_h2e[0].weight, 'grad')
    assert hasattr(lstm_1997.fc_h2e[0].bias, 'grad')

    assert isinstance(batch_cur_states, tuple)
    assert len(batch_cur_states) == 2
    assert isinstance(batch_cur_states[0], list)
    assert len(batch_cur_states[0]) == lstm_1997.n_lyr
    assert isinstance(batch_cur_states[1], list)
    assert len(batch_cur_states[1]) == lstm_1997.n_lyr

    for lyr in range(lstm_1997.n_lyr):
      assert isinstance(batch_cur_states[0][lyr], torch.Tensor)
      assert batch_cur_states[0][lyr].size() == torch.Size([B, lstm_1997.n_blk, lstm_1997.d_blk])
      assert batch_cur_states[0][lyr].dtype == torch.float
      assert not batch_cur_states[0][lyr].requires_grad
      assert isinstance(batch_cur_states[1][lyr], torch.Tensor)
      assert batch_cur_states[1][lyr].size() == torch.Size([B, lstm_1997.d_hid])
      assert batch_cur_states[1][lyr].dtype == torch.float
      assert not batch_cur_states[1][lyr].requires_grad

    lstm_1997.zero_grad()
    batch_prev_states = batch_cur_states
