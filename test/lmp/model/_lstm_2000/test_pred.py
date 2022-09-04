"""Test prediction.

Test target:
- :py:meth:`lmp.model._lstm_2000.LSTM2000.pred`.
"""

import torch

from lmp.model._lstm_2000 import LSTM2000


def test_prediction_result(lstm_2000: LSTM2000, batch_cur_tkids: torch.Tensor) -> None:
  """Return float tensor with correct shape and range."""
  lstm_2000 = lstm_2000.eval()
  seq_len = batch_cur_tkids.size(1)

  batch_prev_states = None
  for idx in range(seq_len):
    batch_next_tkids_pd, batch_cur_states = lstm_2000.pred(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(batch_next_tkids_pd, torch.Tensor)
    assert batch_next_tkids_pd.size() == torch.Size([batch_cur_tkids.shape[0], 1, lstm_2000.emb.num_embeddings])
    assert batch_next_tkids_pd.dtype == torch.float

    # Probabilities are values within range [0, 1].
    assert torch.all(0 <= batch_next_tkids_pd).item()
    assert torch.all(batch_next_tkids_pd <= 1).item()

    # Sum of the probabilities equals to 1.
    accum = batch_next_tkids_pd.sum(dim=-1)
    assert torch.allclose(accum, torch.ones_like(accum))

    assert isinstance(batch_cur_states, tuple)
    assert len(batch_cur_states) == 2
    assert isinstance(batch_cur_states[0], list)
    assert len(batch_cur_states[0]) == lstm_2000.n_lyr
    assert isinstance(batch_cur_states[1], list)
    assert len(batch_cur_states[1]) == lstm_2000.n_lyr

    for lyr in range(lstm_2000.n_lyr):
      c = batch_cur_states[0][lyr]
      h = batch_cur_states[1][lyr]

      assert isinstance(c, torch.Tensor)
      assert c.size() == torch.Size([batch_cur_tkids.size(0), lstm_2000.n_blk, lstm_2000.d_blk])
      assert c.dtype == torch.float
      assert isinstance(h, torch.Tensor)
      assert h.size() == torch.Size([batch_cur_tkids.size(0), lstm_2000.d_hid])
      assert h.dtype == torch.float

    batch_prev_states = batch_cur_states
