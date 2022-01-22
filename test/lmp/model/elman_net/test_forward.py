"""Test forward pass of :py:class:`lmp.model.ElmanNet`.

Test target:
- :py:meth:`lmp.model.ElmanNet.forward`.
"""

import torch

from lmp.model import ElmanNet


def test_forward_path(
  elman_net: ElmanNet,
  batch_cur_tkids: torch.Tensor,
  batch_next_tkids: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  elman_net = elman_net.train()
  elman_net.zero_grad()

  loss = elman_net(batch_cur_tkids=batch_cur_tkids, batch_next_tkids=batch_next_tkids)
  loss.backward()

  assert loss.size() == torch.Size([])
  assert loss.dtype == torch.float
  assert hasattr(elman_net.emb.weight, 'grad')
  assert hasattr(elman_net.h_0, 'grad')
  assert hasattr(elman_net.proj_e2h.weight, 'grad')
  assert hasattr(elman_net.proj_e2h.bias, 'grad')
  assert hasattr(elman_net.proj_h2h.weight, 'grad')
