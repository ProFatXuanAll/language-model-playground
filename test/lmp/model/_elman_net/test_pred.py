"""Test prediction.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.pred`.
"""

import torch

from lmp.model._elman_net import ElmanNet


def test_prediction_result(elman_net: ElmanNet, batch_cur_tkids: torch.Tensor) -> None:
  """Return float tensor with correct shape and range."""
  elman_net = elman_net.eval()
  seq_len = batch_cur_tkids.size(1)

  batch_prev_states = None
  for idx in range(seq_len):
    batch_next_tkids_pd, batch_cur_states = elman_net.pred(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(batch_next_tkids_pd, torch.Tensor)
    assert batch_next_tkids_pd.size() == torch.Size([batch_cur_tkids.shape[0], 1, elman_net.emb.num_embeddings])
    assert batch_next_tkids_pd.dtype == torch.float

    # Probabilities are values within range [0, 1].
    assert torch.all(0 <= batch_next_tkids_pd).item()
    assert torch.all(batch_next_tkids_pd <= 1).item()

    # Sum of the probabilities equals to 1.
    accum = batch_next_tkids_pd.sum(dim=-1)
    assert torch.allclose(accum, torch.ones_like(accum))

    assert isinstance(batch_cur_states, list)
    assert len(batch_cur_states) == elman_net.n_lyr

    for lyr in range(elman_net.n_lyr):
      assert isinstance(batch_cur_states[lyr], torch.Tensor)
      assert batch_cur_states[lyr].size() == torch.Size([batch_cur_tkids.size(0), elman_net.d_hid])

    batch_prev_states = batch_cur_states
