r"""Test the rnn model's forward

Test target:
- :py:meth:`lmp.model._rnn.RNNModel.forward`.
"""
import torch

import pytest


@pytest.mark.parametrize(
    "parameters",
    [
        # Test model forward
        #
        # Expect input `(B, S)` and output `(B, S, V)` with same type.
        {
            "test_input":
                torch.tensor([
                    [0, 2, 4, 6],
                    [1, 3, 5, 7]],
                ),
            "expected": torch.zeros(2, 4, 8),
        },
    ]
)
def test_foward(parameters, model):
    r"""Test :py:meth:lmp.model._rnn.RNNModel.forward"""
    forward = model.forward(parameters["test_input"])

    assert forward.shape == parameters["expected"].shape
    assert forward.dtype == parameters["expected"].dtype
