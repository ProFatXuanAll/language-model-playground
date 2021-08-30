r"""Test loss function of RNN language model.

Test target:
- :py:meth:`lmp.model.RNNModel.loss_fn`.
"""

import torch

from lmp.model import RNNModel


def test_input_shape_and_dtype(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
):
    r"""Input tensors must be long tensors and have the same shape.

    Same shape is required since we are using teacher forcing.
    """
    try:
        rnn_model = rnn_model.eval()
        rnn_model.loss_fn(
            batch_prev_tkids=batch_prev_tkids,
            batch_next_tkids=batch_next_tkids,
        )
    except Exception:
        assert False


def test_return_shape_and_dtype(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
):
    r"""Return float tensor with 0 dimension."""
    rnn_model = rnn_model.eval()
    loss = rnn_model.loss_fn(
        batch_prev_tkids=batch_prev_tkids,
        batch_next_tkids=batch_next_tkids,
    )

    # 0 dimension tensor.
    assert loss.shape == torch.Size([])
    # Return float tensor.
    assert loss.dtype == torch.float


def test_back_propagation_path(
    rnn_model: RNNModel,
    batch_prev_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
):
    r"""Gradients with respect to loss must get back propagated."""
    # Make sure model has no gradients.
    rnn_model = rnn_model.train()
    rnn_model.zero_grad()

    rnn_model.loss_fn(
        batch_prev_tkids=batch_prev_tkids,
        batch_next_tkids=batch_next_tkids,
    ).backward()

    assert hasattr(rnn_model.emb.weight.grad, 'grad')
    assert hasattr(rnn_model.pre_hid[0].weight.grad, 'grad')
    assert hasattr(rnn_model.hid.weight_ih_l0.grad, 'grad')
    assert hasattr(rnn_model.post_hid[-1].weight.grad, 'grad')
