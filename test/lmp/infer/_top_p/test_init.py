r"""Test the constructor of :py:class:`lmp.infer._top_p.TopPInfer`.

Test target:
- :py:meth:`lmp.infer._top_p.TopPInfer.infer`.
"""
import pytest

from lmp.infer._top_p import TopPInfer


def test_max_seq_len():
    r"""``max_seq_len`` must be limited in the range from zero to
    ``hard_max_seq``, and ``max_seq_len`` must be an instance of int."""
    # Test case: `max_seq_len` is negative
    for wrong_max_seq_len in [-1, -2, -3]:
        infer = TopPInfer(
            p=1.0,
            max_seq_len=wrong_max_seq_len,
        )
        assert infer.max_seq_len == infer.hard_max_seq_len

    # Test case: `max_seq_len` is more than `hard_max_len`
    for wrong_max_seq_len in [1000, 2000]:
        infer = TopPInfer(
            p=1.0,
            max_seq_len=wrong_max_seq_len,
        )
        assert infer.max_seq_len == infer.hard_max_seq_len

    # Test case: `max_seq_len` is less than or equal to `hard_max_seq_len` and
    # is more than zero.
    for good_max_seq_len in [1, 20]:
        infer = TopPInfer(
            p=1.0,
            max_seq_len=good_max_seq_len,
        )
        assert infer.max_seq_len == good_max_seq_len

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for wrong_seq_len in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = TopPInfer(
                p=1.0,
                max_seq_len=wrong_seq_len,
            )

        assert (
            '`max_seq_len` must be an instance of `int`' in str(excinfo.value)
        )

    # Test case: Correct input.
    for good_seq_len in [0, 1]:
        infer = TopPInfer(
            p=1.0,
            max_seq_len=good_seq_len,
        )

        assert infer.max_seq_len == good_seq_len


def test_p():
    r"""``p`` must be limited in the range from zero to one, and ``p`` must
    be an instance of float."""
    # Test case: `p` is out of range.
    for wrong_p in [-1.0, -0.0, 2.0]:
        with pytest.raises(ValueError) as excinfo:
            infer = TopPInfer(
                p=wrong_p,
                max_seq_len=0,
            )

        assert (
            '`p` must satisfy `0.0 < p <= 1.0`.' in str(excinfo.value)
        )

    # Test case: `p` is more than zero and less than or equal to one.
    for good_p in [0.1, 0.5, 1.0]:
        infer = TopPInfer(
            p=good_p,
            max_seq_len=0,
        )

        assert infer.p == good_p

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        False, True, 2, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for wrong_p in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = TopPInfer(
                p=wrong_p,
                max_seq_len=0,
            )

        assert (
            '`p` must be an instance of `float`.' in str(excinfo.value)
        )
