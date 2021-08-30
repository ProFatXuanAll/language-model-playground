r"""Test loss function of GRU language model.

Test target:
- :py:meth:`lmp.model.GRUModel.loss_fn`.
"""

import torch

from lmp.model import GRUModel


def test_input_shape_and_dtype(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
):
    r"""Input tensors must be long tensors and have the same shape.

    Same shape is required since we are using teacher forcing.
    """
    try:
        gru_model = gru_model.eval()
        gru_model.loss_fn(
            batch_prev_tkids=batch_prev_tkids,
            batch_next_tkids=batch_next_tkids,
        )
    except Exception:
        assert False


def test_return_shape_and_dtype(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
):
    r"""Return float tensor with 0 dimension."""
    gru_model = gru_model.eval()
    loss = gru_model.loss_fn(
        batch_prev_tkids=batch_prev_tkids,
        batch_next_tkids=batch_next_tkids,
    )

    # 0 dimension tensor.
    assert loss.shape == torch.Size([])
    # Return float tensor.
    assert loss.dtype == torch.float


def test_back_propagation_path(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
):
    r"""Gradients with respect to loss must get back propagated."""
    # Make sure model has no gradients.
    gru_model = gru_model.train()
    gru_model.zero_grad()

    gru_model.loss_fn(
        batch_prev_tkids=batch_prev_tkids,
        batch_next_tkids=batch_next_tkids,
    ).backward()

    assert hasattr(gru_model.emb.weight.grad, 'grad')
    assert hasattr(gru_model.pre_hid[0].weight.grad, 'grad')
    assert hasattr(gru_model.hid.weight_ih_l0.grad, 'grad')
    assert hasattr(gru_model.post_hid[-1].weight.grad, 'grad')
