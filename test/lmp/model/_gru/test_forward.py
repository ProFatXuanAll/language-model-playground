"""Test forward pass of GRU language model.

Test target:
- :py:meth:`lmp.model.GRUModel.forward`.
"""

import torch

from lmp.model import GRUModel


def test_input_shape_and_dtype(
  gru_model: GRUModel,
  batch_prev_tkids: torch.Tensor,
):
  """Input must be long tensor."""

  try:
    gru_model = gru_model.eval()
    gru_model(batch_prev_tkids)
  except Exception:
    assert False


def test_return_shape_and_dtype(
  gru_model: GRUModel,
  batch_prev_tkids: torch.Tensor,
):
  """Return float tensor with correct shape."""
  gru_model = gru_model.eval()
  out = gru_model(batch_prev_tkids)

  # Output float tensor.
  assert out.dtype == torch.float

  # Input shape: (B, S).
  # Output shape: (B, S, V).
  assert out.shape == (
    batch_prev_tkids.shape[0],
    batch_prev_tkids.shape[1],
    gru_model.emb.num_embeddings,
  )


def test_forward_path(
  gru_model: GRUModel,
  batch_prev_tkids: torch.Tensor,
):
  """Parameters used during forward must have gradients."""
  # Make sure model has no gradients.
  gru_model = gru_model.train()
  gru_model.zero_grad()

  gru_model(batch_prev_tkids).sum().backward()

  assert hasattr(gru_model.emb.weight.grad, 'grad')
  assert hasattr(gru_model.pre_hid[0].weight.grad, 'grad')
  assert hasattr(gru_model.hid.weight_ih_l0.grad, 'grad')
  assert hasattr(gru_model.post_hid[-1].weight.grad, 'grad')
