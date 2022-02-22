"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_empty_str`.
"""

import pytest

from lmp.util.validate import raise_if_empty_str


def test_not_raise() -> None:
  """Must not raise when ``val`` is not empty :py:class:`str`."""
  raise_if_empty_str(val='test', val_name='test')


def test_raise_when_empty() -> None:
  """Must raise :py:class:`ValueError` when ``val`` is empty :py:class:`str`."""
  with pytest.raises(ValueError) as excinfo:
    raise_if_empty_str(val='', val_name='test')

  assert '`test` must be non-empty `str`.' in str(excinfo.value)
