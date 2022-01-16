"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_is_directory`.
"""

import os

import pytest

import lmp.util.path
from lmp.util.validate import raise_if_is_directory


def test_not_raise() -> None:
  """Must not raise when ``path`` is a file or does not exist."""
  raise_if_is_directory(path=os.path.join(lmp.util.path.PROJECT_ROOT, 'README.rst'))
  raise_if_is_directory(path='')


def test_raise_when_empty() -> None:
  """Must raise when ``path`` is a directory."""
  with pytest.raises(FileExistsError) as excinfo:
    raise_if_is_directory(path=lmp.util.path.PROJECT_ROOT)

  assert f'{lmp.util.path.PROJECT_ROOT} is a directory.' in str(excinfo.value)
