"""Test prediction of LSTM language model.

Test target:
- :py:meth:`lmp.model.LSTMModel.pred`.
"""

import torch

from lmp.model import LSTMModel


def test_input_shape_and_dtype(
  lstm_model: LSTMModel,
  batch_prev_tkids: torch.Tensor,
):
  """Input must be long tensor."""

  try:
    lstm_model = lstm_model.eval()
    lstm_model.pred(batch_prev_tkids)
  except Exception:
    assert False


def test_return_shape_and_dtype(
  lstm_model: LSTMModel,
  batch_prev_tkids: torch.Tensor,
):
  """Return float tensor with correct shape."""
  lstm_model = lstm_model.eval()
  out = lstm_model.pred(batch_prev_tkids)

  # Output float tensor.
  assert out.dtype == torch.float

  # Input shape: (B, S).
  # Output shape: (B, S, V).
  assert out.shape == (
    batch_prev_tkids.shape[0],
    batch_prev_tkids.shape[1],
    lstm_model.emb.num_embeddings,
  )


def test_value_range(
  lstm_model: LSTMModel,
  batch_prev_tkids: torch.Tensor,
):
  """Return values are probabilities."""
  lstm_model = lstm_model.eval()
  out = lstm_model.pred(batch_prev_tkids)

  # Probabilities are values within range [0, 1].
  assert torch.all(0 <= out).item()
  assert torch.all(out <= 1).item()

  # Sum of the probabilities equals to 1.
  accum_out = out.sum(dim=-1)
  assert torch.allclose(accum_out, torch.ones_like(accum_out))
