r"""Test random seed setting utilities.

Test target:
- :py:meth:`lmp.util.rand.set_seed`.
"""

import random

import pytest

import lmp.util.rand


def test_set_seed():
    r"""Setting random seeds."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_seed in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            lmp.util.rand.set_seed(seed=bad_seed)

        assert (
            '`seed` must be an instance of `int`' in str(excinfo.value)
        )

    # Test case: Invalid value.
    for bad_seed in [-1, -2, -3, 0]:
        with pytest.raises(ValueError) as excinfo:
            lmp.util.rand.set_seed(seed=bad_seed)

        assert (
            '`seed` must be bigger than `0`.' in str(excinfo.value)
        )

    # Test case: correct input.
    lmp.util.rand.set_seed(seed=1)
    state1 = random.getstate()
    lmp.util.rand.set_seed(seed=1)
    state2 = random.getstate()
    assert state1 == state2
