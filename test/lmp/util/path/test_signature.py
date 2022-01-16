"""Test :py:mod:`lmp.util.path` signatures."""

import lmp.util.path


def test_module_attribute():
  """Ensure module attribute's signatures."""
  assert isinstance(lmp.util.path.PROJECT_ROOT, str)
  assert isinstance(lmp.util.path.DATA_PATH, str)
  assert isinstance(lmp.util.path.EXP_PATH, str)
  assert isinstance(lmp.util.path.LOG_PATH, str)
