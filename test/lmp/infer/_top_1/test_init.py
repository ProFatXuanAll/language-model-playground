"""Test the construction of :py:class:`lmp.infer.Top1Infer`.

Test target:
- :py:meth:`lmp.infer.Top1Infer.__init__`.
"""

import pytest

from lmp.infer import Top1Infer


def test_max_seq_len():
  """Perform validation on parameter ``max_seq_len``.

    ``max_seq_len`` must be an instance of ``int``, with value ranging from
    ``-1`` to ``Top1Infer.hard_max_seq_len``.
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
      infer = Top1Infer(max_seq_len=bad_max_seq_len)

    assert ('`max_seq_len` must be an instance of `int`' in str(excinfo.value))

  # Test case: Invalid value.
  for bad_max_seq_len in [-2, -3, Top1Infer.hard_max_seq_len + 1]:
    with pytest.raises(ValueError) as excinfo:
      infer = Top1Infer(max_seq_len=bad_max_seq_len)

    assert ('`max_seq_len` must be in the range from 0 to ' + f'{Top1Infer.hard_max_seq_len}.' in str(excinfo.value))

  # Test case: correct input.
  for good_max_seq_len in [-1, 0, 1, Top1Infer.hard_max_seq_len]:
    infer = Top1Infer(max_seq_len=good_max_seq_len)

    if good_max_seq_len == -1:
      assert infer.max_seq_len == Top1Infer.hard_max_seq_len
    else:
      assert infer.max_seq_len == good_max_seq_len
