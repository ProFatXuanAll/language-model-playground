"""Test prediction.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.pred`.
"""

import torch

from lmp.model._lstm_1997 import LSTM1997


def test_prediction_result(lstm_1997: LSTM1997, batch_cur_tkids: torch.Tensor) -> None:
  """Return float tensor with correct shape and range."""
  lstm_1997 = lstm_1997.eval()
  seq_len = batch_cur_tkids.size(1)

  batch_prev_states = None
  for idx in range(seq_len):
    batch_next_tkids_pd, batch_prev_states = lstm_1997.pred(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    # Output float tensor.
    assert batch_next_tkids_pd.dtype == torch.float

    # Shape: (B, 1, V).
    assert batch_next_tkids_pd.size() == torch.Size([batch_cur_tkids.shape[0], 1, lstm_1997.emb.num_embeddings])

    # Probabilities are values within range [0, 1].
    assert torch.all(0 <= batch_next_tkids_pd).item()
    assert torch.all(batch_next_tkids_pd <= 1).item()

    # Sum of the probabilities equals to 1.
    accum = batch_next_tkids_pd.sum(dim=-1)
    assert torch.allclose(accum, torch.ones_like(accum))

    assert isinstance(batch_prev_states, list)
    assert len(batch_prev_states) == 2 * lstm_1997.n_lyr
    for lyr in range(lstm_1997.n_lyr):
      h = batch_prev_states[2 * lyr]
      c = batch_prev_states[2 * lyr + 1]
      assert h.size() == torch.Size([batch_cur_tkids.size(0), lstm_1997.d_hid])
      assert c.size() == torch.Size([batch_cur_tkids.size(0), lstm_1997.n_blk, lstm_1997.d_blk])
