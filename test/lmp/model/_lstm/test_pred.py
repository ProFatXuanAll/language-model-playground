r"""Test the rnn model's prediction

Test target:
- :py:meth:`lmp.model._lstm.LSTMModel.pred`.
"""
import torch

import pytest


@pytest.mark.parametrize(
    "parameters",
    [
        # Test model prediction
        #
        # Expect input `(B, S)` and output `(B, S, V)` which have same type.
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
def test_pred(parameters, model):
    r"""Test :py:meth:lmp.model._lstm.LSTMModel.forward"""
    pred = model.pred(parameters["test_input"])

    assert pred.shape == parameters["expected"].shape
    assert pred.dtype == parameters["expected"].dtype
