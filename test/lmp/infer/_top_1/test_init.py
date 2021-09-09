r"""Test the constructor of :py:class:`lmp.infer._top_1.Top1Infer`.

Test target:
- :py:meth:`lmp.infer._top_1.Top1Infer.infer`.
"""
import pytest

from lmp.infer._top_1 import Top1Infer


def test_max_seq_len():
    r"""``max_seq_len`` must be limited in the range from zero to
    ``hard_max_seq``, and ``max_seq_len`` must be an instance of int."""
    # Test case: Wrong value input
    for wrong_max_seq_len in [-1, -2, -3, 1000, 2000]:
        with pytest.raises(ValueError) as excinfo:
            infer = Top1Infer(
                max_seq_len=wrong_max_seq_len,
            )

        assert (
            '`self.max_seq_len` must be less than or equal to '
            + '`self.hard_max_seq_len` and more than or equal to zero.'
            in str(excinfo.value)
        )

    # Test case: `max_seq_len` is less than or equal to `hard_max_seq_len` and
    # is more than zero.
    for good_max_seq_len in [1, 20]:
        infer = Top1Infer(
            max_seq_len=good_max_seq_len,
        )
        assert infer.max_seq_len == good_max_seq_len

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for wrong_seq_len in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = Top1Infer(
                max_seq_len=wrong_seq_len,
            )

        assert (
            '`max_seq_len` must be an instance of `int`' in str(excinfo.value)
        )

    # Test case: Correct input.
    for good_seq_len in [0, 1]:
        infer = Top1Infer(
            max_seq_len=good_seq_len,
        )

        assert infer.max_seq_len == good_seq_len
