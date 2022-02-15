"""Test :py:mod:`lmp.util.path` signatures."""

import lmp.util.path


def test_module_attribute():
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.util.path, 'PROJECT_ROOT')
  assert isinstance(lmp.util.path.PROJECT_ROOT, str)
  assert hasattr(lmp.util.path, 'DATA_PATH')
  assert isinstance(lmp.util.path.DATA_PATH, str)
  assert hasattr(lmp.util.path, 'EXP_PATH')
  assert isinstance(lmp.util.path.EXP_PATH, str)
  assert hasattr(lmp.util.path, 'LOG_PATH')
  assert isinstance(lmp.util.path.LOG_PATH, str)
