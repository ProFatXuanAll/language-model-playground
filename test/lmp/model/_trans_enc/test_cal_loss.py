"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._trans_enc.TransEnc.cal_loss`.
"""

import torch

from lmp.model._trans_enc import TransEnc


def test_cal_loss(batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor, trans_enc: TransEnc) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  trans_enc = trans_enc.train()
  trans_enc.zero_grad()

  B = batch_cur_tkids.size(0)
  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    loss, batch_cur_states = trans_enc.cal_loss(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_next_tkids=batch_next_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.size() == torch.Size([])
    assert loss.dtype == torch.float

    loss.backward()
    assert hasattr(trans_enc.emb.weight, 'grad')

    for lyr in range(trans_enc.n_lyr):
      assert hasattr(trans_enc.stack_trans_enc[lyr].ffn[0].weight, 'grad')

    assert isinstance(batch_cur_states, torch.Tensor)
    batch_cur_states.size(0) == B
    if batch_prev_states is None:
      assert batch_cur_states.size(1) == 1
    else:
      assert batch_cur_states.size(1) == min(batch_prev_states.size(1) + 1, trans_enc.max_seq_len - 1)

    trans_enc.zero_grad()
    batch_prev_states = batch_cur_states
