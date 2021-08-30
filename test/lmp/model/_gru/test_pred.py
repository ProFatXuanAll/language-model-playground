r"""Test prediction of GRU language model.

Test target:
- :py:meth:`lmp.model.GRUModel.pred`.
"""

import torch

from lmp.model import GRUModel


def test_input_shape_and_dtype(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
):
    r"""Input must be long tensor."""

    try:
        gru_model = gru_model.eval()
        gru_model.pred(batch_prev_tkids)
    except Exception:
        assert False


def test_return_shape_and_dtype(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
):
    r"""Return float tensor with correct shape."""
    gru_model = gru_model.eval()
    out = gru_model.pred(batch_prev_tkids)

    # Output float tensor.
    assert out.dtype == torch.float

    # Input shape: (B, S).
    # Output shape: (B, S, V).
    assert out.shape == (
        batch_prev_tkids.shape[0],
        batch_prev_tkids.shape[1],
        gru_model.emb.num_embeddings,
    )


def test_value_range(
    gru_model: GRUModel,
    batch_prev_tkids: torch.Tensor,
):
    r"""Return values are probabilities."""
    gru_model = gru_model.eval()
    out = gru_model.pred(batch_prev_tkids)

    # Probabilities are values within range [0, 1].
    assert torch.all(0 <= out).item()
    assert torch.all(out <= 1).item()

    # Sum of the probabilities equals to 1.
    accum_out = out.sum(dim=-1)
    assert torch.allclose(accum_out, torch.ones_like(accum_out))
