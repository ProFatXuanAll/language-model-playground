r"""Test the constructor of :py:class:`lmp.infer._top_k.TopKInfer`.

Test target:
- :py:meth:`lmp.infer._top_k.TopKInfer.infer`.
"""
import pytest

from lmp.infer._top_k import TopKInfer


def test_max_seq_len():
    r"""``max_seq_len`` must be limited in the range from zero to
    ``hard_max_seq``, and ``max_seq_len`` must be an instance of int."""
    # Test case: Wrong value input
    for wrong_max_seq_len in [-1, -2, -3, 1000, 2000]:
        with pytest.raises(ValueError) as excinfo:
            infer = TopKInfer(
                k=1,
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
        infer = TopKInfer(
            k=1,
            max_seq_len=good_max_seq_len,
        )
        assert infer.max_seq_len == good_max_seq_len

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for wrong_seq_len in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = TopKInfer(
                k=1,
                max_seq_len=wrong_seq_len,
            )

        assert (
            '`max_seq_len` must be an instance of `int`' in str(excinfo.value)
        )

    # Test case: Correct input.
    for good_seq_len in [0, 1]:
        infer = TopKInfer(
            k=1,
            max_seq_len=good_seq_len,
        )

        assert infer.max_seq_len == good_seq_len


def test_k():
    r"""``k`` must more than zero, and ``max_seq_len`` must be an instance
    of int."""
    # Test case: `k` is negative.
    for wrong_k in [0, -1, -2]:
        with pytest.raises(ValueError) as excinfo:
            infer = TopKInfer(
                k=wrong_k,
                max_seq_len=0,
            )

        assert (
            '`k` must satisfy `k > 0`.' in str(excinfo.value)
        )

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for wrong_k in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = TopKInfer(
                k=wrong_k,
                max_seq_len=0,
            )

        assert (
            '`k` must be an instance of `int`.' in str(excinfo.value)
        )

    # Test case: Correct input.
    for good_k in [1, 2]:
        infer = TopKInfer(
            k=good_k,
            max_seq_len=0,
        )

        assert infer.k == good_k
