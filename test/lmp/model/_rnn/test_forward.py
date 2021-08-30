r"""Test forward pass of RNN language model.

Test target:
- :py:meth:`lmp.model.RNNModel.forward`.
"""

import torch

from lmp.model import RNNModel


def test_input_shape_and_dtype(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
):
    r"""Input must be long tensor."""

    try:
        rnn_model = rnn_model.eval()
        rnn_model(batch_prev_tkids)
    except Exception:
        assert False


def test_return_shape_and_dtype(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
):
    r"""Return float tensor with correct shape."""
    rnn_model = rnn_model.eval()
    out = rnn_model(batch_prev_tkids)

    # Output float tensor.
    assert out.dtype == torch.float

    # Input shape: (B, S).
    # Output shape: (B, S, V).
    assert out.shape == (
        batch_prev_tkids.shape[0],
        batch_prev_tkids.shape[1],
        rnn_model.emb.num_embeddings,
    )


def test_forward_path(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
):
    r"""Parameters used during forward must have gradients."""
    # Make sure model has no gradients.
    rnn_model = rnn_model.train()
    rnn_model.zero_grad()

    rnn_model(batch_prev_tkids).sum().backward()

    assert hasattr(rnn_model.emb.weight.grad, 'grad')
    assert hasattr(rnn_model.pre_hid[0].weight.grad, 'grad')
    assert hasattr(rnn_model.hid.weight_ih_l0.grad, 'grad')
    assert hasattr(rnn_model.post_hid[-1].weight.grad, 'grad')
