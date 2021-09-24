r"""Test setting the random seed.

Test target:
- :py:meth:`lmp.util.rand.set_seed`.
"""

import pytest

import lmp.util.rand


def test_set_seed():
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
            '`seed` must bigger than `0`.' in str(excinfo.value)
        )

    # Test case: correct input.
    for good_seed in [1, 2]:
        lmp.util.rand.set_seed(seed=good_seed)
