r"""Test the construction of :py:class:`lmp.infer.TopKInfer`.

Test target:
- :py:meth:`lmp.infer.TopKInfer.__init__`.
"""

import pytest

from lmp.infer import TopKInfer


def test_max_seq_len():
    r"""Perform validation on parameter ``max_seq_len``.

    ``max_seq_len`` must be an instance of ``int``, with value ranging from
    ``0`` to ``TopKInfer.hard_max_seq_len``.
    """
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_max_seq_len in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = TopKInfer(
                k=1,
                max_seq_len=bad_max_seq_len,
            )

        assert (
            '`max_seq_len` must be an instance of `int`' in str(excinfo.value)
        )

    # Test case: Invalid value.
    for bad_max_seq_len in [-1, -2, -3, TopKInfer.hard_max_seq_len + 1]:
        with pytest.raises(ValueError) as excinfo:
            infer = TopKInfer(
                k=1,
                max_seq_len=bad_max_seq_len,
            )

        assert (
            '`max_seq_len` must be in the range from 0 to '
            + f'{TopKInfer.hard_max_seq_len}.'
            in str(excinfo.value)
        )

    # Test case: correct input.
    for good_max_seq_len in [0, 1, TopKInfer.hard_max_seq_len]:
        infer = TopKInfer(
            k=1,
            max_seq_len=good_max_seq_len,
        )

        assert infer.max_seq_len == good_max_seq_len


def test_k():
    r"""``k`` must be an instance of ``int`` and larger than ``0``."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_k in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            infer = TopKInfer(
                k=bad_k,
                max_seq_len=0,
            )

        assert '`k` must be an instance of `int`.' in str(excinfo.value)

    # Test case: Invalid value.
    for bad_k in [0, -1, -2]:
        with pytest.raises(ValueError) as excinfo:
            infer = TopKInfer(
                k=bad_k,
                max_seq_len=0,
            )

        assert '`k` must satisfy `k > 0`' in str(excinfo.value)

    # Test case: Correct input.
    for good_k in [1, 2]:
        infer = TopKInfer(
            k=good_k,
            max_seq_len=0,
        )

        assert infer.k == good_k
