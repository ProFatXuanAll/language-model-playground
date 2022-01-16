"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_not_in`.
"""

import pytest

from lmp.util.validate import raise_if_not_in


def test_not_raise() -> None:
  """Must not raise when ``val`` is in ``val_range``."""
  val_range = [False, True, 0, 1, -1, 0.1, -0.1, 'test', None]
  for val in val_range:
    raise_if_not_in(val=val, val_name='test', val_range=val_range)


def test_raise_if_not_in() -> None:
  """Must raise :py:class:`ValueError` when ``val`` is not in ``val_range``."""
  val_range = [False, True, 0, 1, -1, 0.1, -0.1, 'test', None]
  with pytest.raises(ValueError) as excinfo:
    raise_if_not_in(val='', val_name='test', val_range=val_range)

  assert '`test` must be one of the following values:' in str(excinfo.value)
  for val_str in map(str, val_range):
    assert val_str in str(excinfo.value)
