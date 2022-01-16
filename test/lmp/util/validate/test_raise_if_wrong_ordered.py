"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_wrong_ordered`.
"""

import pytest

from lmp.util.validate import raise_if_wrong_ordered


def test_not_raise() -> None:
  """Must not raise when list values are ascending."""
  for vals in [
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, 1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (-0.1, -0.1, -0.1),
    (-0.1, -0.1, 0.0),
    (-0.1, -0.1, 0.1),
    (-0.1, 0.0, 0.0),
    (-0.1, 0.0, 0.1),
    (-0.1, 0.1, 0.1),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.1),
    (0.0, 0.1, 0.1),
    (0.1, 0.1, 0.1),
  ]:
    raise_if_wrong_ordered(vals=vals, val_names=['a', 'b', 'c'])


def test_raise_if_wrong_ordered() -> None:
  """Must raise :py:class:`ValueError` when list values are not ascending."""
  for vals in [
    (-1, 0, -1),
    (-1, 1, -1),
    (-1, 1, 0),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
    (0, 1, -1),
    (0, 1, 0),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (-0.1, 0.0, -0.1),
    (-0.1, 0.1, -0.1),
    (-0.1, 0.1, 0.0),
    (0.0, -0.1, -0.1),
    (0.0, -0.1, 0.0),
    (0.0, -0.1, 0.1),
    (0.0, 0.0, -0.1),
    (0.0, 0.1, -0.1),
    (0.0, 0.1, 0.0),
    (0.1, -0.1, -0.1),
    (0.1, -0.1, 0.0),
    (0.1, -0.1, 0.1),
    (0.1, 0.0, -0.1),
    (0.1, 0.0, 0.0),
    (0.1, 0.0, 0.1),
    (0.1, 0.1, -0.1),
    (0.1, 0.1, 0.0),
  ]:
    with pytest.raises(ValueError) as excinfo:
      raise_if_wrong_ordered(vals=vals, val_names=['a', 'b', 'c'])

    assert 'Must have `a <= b <= c`.' in str(excinfo.value)
