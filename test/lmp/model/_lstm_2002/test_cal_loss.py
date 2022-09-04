"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._lstm_2002.LSTM2002.cal_loss`.
"""

import torch

from lmp.model._lstm_2002 import LSTM2002


def test_cal_loss(lstm_2002: LSTM2002, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  lstm_2002 = lstm_2002.train()
  lstm_2002.zero_grad()

  B = batch_cur_tkids.size(0)
  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    loss, batch_cur_states = lstm_2002.cal_loss(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_next_tkids=batch_next_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float

    loss.backward()
    assert hasattr(lstm_2002.emb.weight, 'grad')
    assert hasattr(lstm_2002.fc_e2h[1].weight, 'grad')
    assert hasattr(lstm_2002.fc_e2h[1].bias, 'grad')

    for lyr in range(lstm_2002.n_lyr):
      assert hasattr(lstm_2002.stack_rnn[2 * lyr].fc_x2ig.weight, 'grad')

    assert hasattr(lstm_2002.fc_h2e[0].weight, 'grad')
    assert hasattr(lstm_2002.fc_h2e[0].bias, 'grad')

    assert isinstance(batch_cur_states, tuple)
    assert len(batch_cur_states) == 2
    assert isinstance(batch_cur_states[0], list)
    assert len(batch_cur_states[0]) == lstm_2002.n_lyr
    assert isinstance(batch_cur_states[1], list)
    assert len(batch_cur_states[1]) == lstm_2002.n_lyr

    for lyr in range(lstm_2002.n_lyr):
      assert isinstance(batch_cur_states[0][lyr], torch.Tensor)
      assert batch_cur_states[0][lyr].size() == torch.Size([B, lstm_2002.n_blk, lstm_2002.d_blk])
      assert not batch_cur_states[0][lyr].requires_grad
      assert isinstance(batch_cur_states[1][lyr], torch.Tensor)
      assert batch_cur_states[1][lyr].size() == torch.Size([B, lstm_2002.d_hid])
      assert not batch_cur_states[1][lyr].requires_grad

    lstm_2002.zero_grad()
    batch_prev_states = batch_cur_states
