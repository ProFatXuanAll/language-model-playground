"""Test path hierarchy.

Test target:
- :py:attr:`lmp.vars.PROJECT_ROOT`.
- :py:attr:`lmp.vars.DATA_PATH`.
- :py:attr:`lmp.vars.EXP_PATH`.
- :py:attr:`lmp.vars.LOG_PATH`.
"""

import os

import lmp.vars


def test_absolute_path():
  """All path are absolute."""
  assert os.path.isabs(lmp.vars.PROJECT_ROOT)
  assert os.path.isabs(lmp.vars.DATA_PATH)
  assert os.path.isabs(lmp.vars.EXP_PATH)
  assert os.path.isabs(lmp.vars.LOG_PATH)


def test_sub_path():
  """All path are subpath to :py:attr:`lmp.vars.PROJECT_ROOT`."""
  assert os.path.commonpath([
    lmp.vars.DATA_PATH,
    lmp.vars.EXP_PATH,
    lmp.vars.LOG_PATH,
  ]) == lmp.vars.PROJECT_ROOT
