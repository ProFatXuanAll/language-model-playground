"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.forward`.
- :py:meth:`lmp.model._elman_net.ElmanNetLayer.forward`.
"""

import torch

from lmp.model._elman_net import ElmanNet


def test_elman_net_layer_forward_path(
  elman_net_layer: ElmanNet,
  batch_x: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  elman_net_layer = elman_net_layer.train()
  elman_net_layer.zero_grad()

  batch_prev_states = None
  for idx in range(batch_x.size(1)):
    batch_cur_states = elman_net_layer(
      batch_x=batch_x[:, idx, :].unsqueeze(1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(batch_cur_states, list)
    assert len(batch_cur_states) == 1

    batch_prev_states = [batch_cur_states[0].detach()[:, -1, :]]

    loss = batch_cur_states[0].sum()
    loss.backward()

    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float
    assert hasattr(elman_net_layer.fc_x2h.weight, 'grad')
    assert hasattr(elman_net_layer.fc_x2h.bias, 'grad')
    assert hasattr(elman_net_layer.fc_h2h.weight, 'grad')
    assert hasattr(elman_net_layer.h_0, 'grad')

    if idx == 0:
      assert hasattr(elman_net_layer.h_0, 'grad')

    elman_net_layer.zero_grad()


def test_elman_net_forward_path(
  elman_net: ElmanNet,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  elman_net = elman_net.train()
  elman_net.zero_grad()

  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    loss, batch_prev_states = elman_net.loss(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_next_tkids=batch_next_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )
    loss.backward()

    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float
    assert hasattr(elman_net.emb.weight, 'grad')
    assert hasattr(elman_net.fc_e2h[1].weight, 'grad')
    assert hasattr(elman_net.fc_e2h[1].bias, 'grad')
    for lyr in range(elman_net.n_lyr):
      assert hasattr(elman_net.stack_rnn[2 * lyr].fc_x2h.weight, 'grad')
    assert hasattr(elman_net.fc_h2e[0].weight, 'grad')
    assert hasattr(elman_net.fc_h2e[0].bias, 'grad')

    elman_net.zero_grad()
