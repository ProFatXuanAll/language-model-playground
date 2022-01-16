"""Test the construction of :py:class:`lmp.infer.TopPInfer`.

Test target:
- :py:meth:`lmp.infer.TopPInfer.__init__`.
"""

import pytest

from lmp.infer import TopPInfer


def test_max_seq_len():
  """Perform validation on parameter ``max_seq_len``.

    ``max_seq_len`` must be an instance of ``int``, with value ranging from
    ``-1`` to ``TopPInfer.hard_max_seq_len``.
    """
  # Test case: Type mismatched.
  wrong_typed_inputs = [
    0.0,
    0.1,
    1.0,
    '',
    (),
    [],
    {},
    set(),
    None,
    ...,
    NotImplemented,
  ]

  for bad_max_seq_len in wrong_typed_inputs:
    with pytest.raises(TypeError) as excinfo:
      infer = TopPInfer(
        max_seq_len=bad_max_seq_len,
        p=1.0,
      )

    assert ('`max_seq_len` must be an instance of `int`' in str(excinfo.value))

  # Test case: Invalid value.
  for bad_max_seq_len in [-2, -3, TopPInfer.hard_max_seq_len + 1]:
    with pytest.raises(ValueError) as excinfo:
      infer = TopPInfer(
        max_seq_len=bad_max_seq_len,
        p=1.0,
      )

    assert ('`max_seq_len` must be in the range from 0 to ' + f'{TopPInfer.hard_max_seq_len}.' in str(excinfo.value))

  # Test case: correct input.
  for good_max_seq_len in [-1, 0, 1, TopPInfer.hard_max_seq_len]:
    infer = TopPInfer(
      max_seq_len=good_max_seq_len,
      p=1.0,
    )

    if good_max_seq_len == -1:
      assert infer.max_seq_len == TopPInfer.hard_max_seq_len
    else:
      assert infer.max_seq_len == good_max_seq_len


def test_p():
  """Perform validation on parameter ``p``.

    ``p`` must be an instance of ``float`` with range of non-zero probability.
    """

  # Test case: Type mismatched.
  wrong_typed_inputs = [
    False,
    True,
    0,
    1,
    '',
    (),
    [],
    {},
    set(),
    None,
    ...,
    NotImplemented,
  ]

  for bad_p in wrong_typed_inputs:
    with pytest.raises(TypeError) as excinfo:
      infer = TopPInfer(
        p=bad_p,
        max_seq_len=0,
      )

    assert '`p` must be an instance of `float`.' in str(excinfo.value)

  # Test case: Invalid value.
  for bad_p in [-1.0, -0.1, 0.0, 1.1, 2.0]:
    with pytest.raises(ValueError) as excinfo:
      infer = TopPInfer(
        p=bad_p,
        max_seq_len=0,
      )

    assert '`p` must satisfy `0.0 < p <= 1.0`.' in str(excinfo.value)

  # Test case: Correct input.
  for good_p in [0.1, 0.5, 1.0]:
    infer = TopPInfer(
      p=good_p,
      max_seq_len=0,
    )

    assert infer.p == good_p
