"""Test prediction.

Test target:
- :py:meth:`lmp.model._trans_enc.TransEnc.pred`.
"""

import torch

from lmp.model._trans_enc import TransEnc


def test_prediction_result(batch_cur_tkids: torch.Tensor, trans_enc: TransEnc) -> None:
  """Return float tensor with correct shape and range."""
  trans_enc = trans_enc.eval()
  B = batch_cur_tkids.size(0)
  seq_len = batch_cur_tkids.size(1)

  batch_prev_states = None
  for idx in range(seq_len):
    batch_next_tkids_pd, batch_cur_states = trans_enc.pred(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(batch_next_tkids_pd, torch.Tensor)
    assert batch_next_tkids_pd.size() == torch.Size([batch_cur_tkids.shape[0], 1, trans_enc.emb.num_embeddings])
    assert batch_next_tkids_pd.dtype == torch.float

    # Probabilities are values within range [0, 1].
    assert torch.all(0 <= batch_next_tkids_pd).item()
    assert torch.all(batch_next_tkids_pd <= 1).item()

    # Sum of the probabilities equals to 1.
    accum = batch_next_tkids_pd.sum(dim=-1)
    assert torch.allclose(accum, torch.ones_like(accum))

    assert isinstance(batch_cur_states, torch.Tensor)
    batch_cur_states.size(0) == B
    if batch_prev_states is None:
      assert batch_cur_states.size(1) == 1
    else:
      assert batch_cur_states.size(1) == min(batch_prev_states.size(1) + 1, trans_enc.max_seq_len - 1)

    batch_prev_states = batch_cur_states
