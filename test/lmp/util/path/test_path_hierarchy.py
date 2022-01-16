"""Test path hierarchy.

Test target:
- :py:attr:`lmp.util.path.PROJECT_ROOT`.
- :py:attr:`lmp.util.path.DATA_PATH`.
- :py:attr:`lmp.util.path.EXP_PATH`.
- :py:attr:`lmp.util.path.LOG_PATH`.
"""

import os

import lmp.util.path


def test_absolute_path():
  """All path are absolute."""
  assert os.path.isabs(lmp.util.path.PROJECT_ROOT)
  assert os.path.isabs(lmp.util.path.DATA_PATH)
  assert os.path.isabs(lmp.util.path.EXP_PATH)
  assert os.path.isabs(lmp.util.path.LOG_PATH)


def test_sub_path():
  """All path are subpath to :py:attr:`lmp.util.path.PROJECT_ROOT`."""
  assert os.path.commonpath(
    [
      lmp.util.path.DATA_PATH,
      lmp.util.path.EXP_PATH,
      lmp.util.path.LOG_PATH,
    ]
  ) == lmp.util.path.PROJECT_ROOT
