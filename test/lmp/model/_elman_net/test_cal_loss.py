"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.cal_loss`.
"""

import torch

from lmp.model._elman_net import ElmanNet


def test_cal_loss(elman_net: ElmanNet, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  elman_net = elman_net.train()
  elman_net.zero_grad()

  B = batch_cur_tkids.size(0)
  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    loss, batch_cur_states = elman_net.cal_loss(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_next_tkids=batch_next_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float

    loss.backward()
    assert hasattr(elman_net.emb.weight, 'grad')
    assert hasattr(elman_net.fc_e2h[1].weight, 'grad')
    assert hasattr(elman_net.fc_e2h[1].bias, 'grad')

    for lyr in range(elman_net.n_lyr):
      assert hasattr(elman_net.stack_rnn[2 * lyr].fc_x2h.weight, 'grad')

    assert hasattr(elman_net.fc_h2e[0].weight, 'grad')
    assert hasattr(elman_net.fc_h2e[0].bias, 'grad')

    assert isinstance(batch_cur_states, list)
    assert len(batch_cur_states) == elman_net.n_lyr

    for lyr in range(elman_net.n_lyr):
      assert isinstance(batch_cur_states[lyr], torch.Tensor)
      assert batch_cur_states[lyr].size() == torch.Size([B, elman_net.d_hid])
      assert not batch_cur_states[lyr].requires_grad

    elman_net.zero_grad()
    batch_prev_states = batch_cur_states
