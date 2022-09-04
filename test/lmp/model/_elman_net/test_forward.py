"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.forward`.
- :py:meth:`lmp.model._elman_net.ElmanNetLayer.forward`.
"""

import torch

from lmp.model._elman_net import ElmanNet, ElmanNetLayer


def test_elman_net_layer_forward_path(elman_net_layer: ElmanNetLayer, x: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  elman_net_layer = elman_net_layer.train()
  elman_net_layer.zero_grad()

  B = x.size(0)
  h_0 = None
  for idx in range(x.size(1)):
    h = elman_net_layer(x=x[:, idx, :].unsqueeze(1), h_0=h_0)

    assert isinstance(h, torch.Tensor)
    assert h.size() == torch.Size([B, 1, elman_net_layer.out_feat])

    logits = h.sum()
    logits.backward()

    assert logits.size() == torch.Size([])
    assert logits.dtype == torch.float
    assert hasattr(elman_net_layer.fc_x2h.weight, 'grad')
    assert hasattr(elman_net_layer.fc_x2h.bias, 'grad')
    assert hasattr(elman_net_layer.fc_h2h.weight, 'grad')

    elman_net_layer.zero_grad()
    h_0 = h.detach()[:, -1, :]


def test_elman_net_forward_path(elman_net: ElmanNet, batch_cur_tkids: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  elman_net = elman_net.train()
  elman_net.zero_grad()

  B = batch_cur_tkids.size(0)
  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    logits, batch_cur_states = elman_net(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(logits, torch.Tensor)
    assert logits.size() == torch.Size([B, 1, elman_net.emb.num_embeddings])
    assert logits.dtype == torch.float

    logits.sum().backward()
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
