"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_is_file`.
"""

import os

import pytest

import lmp.util.path
from lmp.util.validate import raise_if_is_file


def test_not_raise() -> None:
  """Must not raise when ``path`` is a directory or does not exist."""
  raise_if_is_file(path=lmp.util.path.PROJECT_ROOT)
  raise_if_is_file(path='')


def test_raise_when_empty() -> None:
  """Must raise when ``path`` is a file."""
  path = os.path.join(lmp.util.path.PROJECT_ROOT, 'README.rst')
  with pytest.raises(FileExistsError) as excinfo:
    raise_if_is_file(path=path)

  assert f'{path} is a file.' in str(excinfo.value)
