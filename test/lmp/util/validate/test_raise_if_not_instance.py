"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_not_instance`.
"""

import pytest

from lmp.util.validate import raise_if_not_instance


def test_not_raise() -> None:
  """Must not raise when ``val`` is instance of ``val_type``."""
  for val in [False, True, 0, 1, -1, 0.1, -0.1, '', None, (), [], {}, set(), ..., NotImplemented]:
    raise_if_not_instance(val=val, val_name='test', val_type=type(val))


def test_raise_if_not_int() -> None:
  """Must raise :py:class:`TypeError` when ``val`` is not instance of :py:class:`int`."""
  for not_int in [0.1, -0.1, '', None, (), [], {}, set(), ..., NotImplemented]:
    with pytest.raises(TypeError) as excinfo:
      raise_if_not_instance(val=not_int, val_name='test', val_type=int)

    assert '`test` must be an instance of `int`.' in str(excinfo.value)


def test_raise_if_not_float() -> None:
  """Must raise :py:class:`TypeError` when ``val`` is not instance of :py:class:`float`."""
  for not_float in [0, 1, -1, '', None, (), [], {}, set(), ..., NotImplemented]:
    with pytest.raises(TypeError) as excinfo:
      raise_if_not_instance(val=not_float, val_name='test', val_type=float)

    assert '`test` must be an instance of `float`.' in str(excinfo.value)


def test_raise_if_not_str() -> None:
  """Must raise :py:class:`TypeError` when ``val`` is not instance of :py:class:`str`."""
  for not_str in [False, True, 0, 1, -1, 0.1, -0.1, None, (), [], {}, set(), ..., NotImplemented]:
    with pytest.raises(TypeError) as excinfo:
      raise_if_not_instance(val=not_str, val_name='test', val_type=str)

    assert '`test` must be an instance of `str`.' in str(excinfo.value)


def test_raise_if_not_bool() -> None:
  """Must raise :py:class:`TypeError` when ``val`` is not instance of :py:class:`bool`."""
  for not_bool in [0, 1, -1, 0.1, -0.1, '', None, (), [], {}, set(), ..., NotImplemented]:
    with pytest.raises(TypeError) as excinfo:
      raise_if_not_instance(val=not_bool, val_name='test', val_type=bool)

    assert '`test` must be an instance of `bool`.' in str(excinfo.value)
