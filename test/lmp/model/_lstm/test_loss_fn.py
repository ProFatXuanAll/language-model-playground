r"""Test the rnn model's loss function

Test target:
- :py:meth:`lmp.model._lstm.LSTMModel.loss_fn`.
"""
import torch

import pytest


@pytest.mark.parametrize(
    "parameters",
    [
        # Test cross entropy
        #
        # Expect input  prev_tkids `(BxS, V)` and next_tkids `(BxS)` and
        # output `(1)`.
        {
            "prev_tkids":
                torch.tensor([
                    [0, 2, 4, 6],
                    [1, 3, 5, 7]],
                ),
            "next_tkids":
                torch.tensor([
                    [0, 2, 4, 6],
                    [1, 3, 5, 7],
                ]),
        },
    ]
)
def test_loss_fn(parameters, model):
    r"""Test :py:meth:lmp.model._lstm.LSTMModel.forward"""
    loss = model.loss_fn(parameters["next_tkids"], parameters["prev_tkids"])

    assert loss.dtype == torch.float
