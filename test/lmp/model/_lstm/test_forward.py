"""Test forward pass of LSTM language model.

Test target:
- :py:meth:`lmp.model.LSTMModel.forward`.
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
    lstm_model(batch_prev_tkids)
  except Exception:
    assert False


def test_return_shape_and_dtype(
  lstm_model: LSTMModel,
  batch_prev_tkids: torch.Tensor,
):
  """Return float tensor with correct shape."""
  lstm_model = lstm_model.eval()
  out = lstm_model(batch_prev_tkids)

  # Output float tensor.
  assert out.dtype == torch.float

  # Input shape: (B, S).
  # Output shape: (B, S, V).
  assert out.shape == (
    batch_prev_tkids.shape[0],
    batch_prev_tkids.shape[1],
    lstm_model.emb.num_embeddings,
  )


def test_forward_path(
  lstm_model: LSTMModel,
  batch_prev_tkids: torch.Tensor,
):
  """Parameters used during forward must have gradients."""
  # Make sure model has no gradients.
  lstm_model = lstm_model.train()
  lstm_model.zero_grad()

  lstm_model(batch_prev_tkids).sum().backward()

  assert hasattr(lstm_model.emb.weight.grad, 'grad')
  assert hasattr(lstm_model.pre_hid[0].weight.grad, 'grad')
  assert hasattr(lstm_model.hid.weight_ih_l0.grad, 'grad')
  assert hasattr(lstm_model.post_hid[-1].weight.grad, 'grad')
