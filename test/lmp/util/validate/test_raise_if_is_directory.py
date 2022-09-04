"""Test raise exception.

Test target:
- :py:meth:`lmp.util.validate.raise_if_is_directory`.
"""

import os

import pytest

import lmp.vars
from lmp.util.validate import raise_if_is_directory


def test_not_raise() -> None:
  """Must not raise when ``path`` is a file or does not exist."""
  raise_if_is_directory(path=os.path.join(lmp.vars.PROJECT_ROOT, 'README.rst'))
  raise_if_is_directory(path='')


def test_raise_when_empty() -> None:
  """Must raise when ``path`` is a directory."""
  with pytest.raises(FileExistsError) as excinfo:
    raise_if_is_directory(path=lmp.vars.PROJECT_ROOT)

  assert f'{lmp.vars.PROJECT_ROOT} is a directory.' in str(excinfo.value)
