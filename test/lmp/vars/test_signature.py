"""Test :py:mod:`lmp.vars` signatures."""

import lmp.vars


def test_module_attribute():
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.vars, 'BOS_TK')
  assert isinstance(lmp.vars.BOS_TK, str)
  assert hasattr(lmp.vars, 'BOS_TKID')
  assert isinstance(lmp.vars.BOS_TKID, int)
  assert hasattr(lmp.vars, 'DATA_PATH')
  assert isinstance(lmp.vars.DATA_PATH, str)
  assert hasattr(lmp.vars, 'EOS_TK')
  assert isinstance(lmp.vars.EOS_TK, str)
  assert hasattr(lmp.vars, 'EOS_TKID')
  assert isinstance(lmp.vars.EOS_TKID, int)
  assert hasattr(lmp.vars, 'EXP_PATH')
  assert isinstance(lmp.vars.EXP_PATH, str)
  assert hasattr(lmp.vars, 'LOG_PATH')
  assert isinstance(lmp.vars.LOG_PATH, str)
  assert hasattr(lmp.vars, 'PAD_TK')
  assert isinstance(lmp.vars.PAD_TK, str)
  assert hasattr(lmp.vars, 'PAD_TKID')
  assert isinstance(lmp.vars.PAD_TKID, int)
  assert hasattr(lmp.vars, 'PROJECT_ROOT')
  assert isinstance(lmp.vars.PROJECT_ROOT, str)
  assert hasattr(lmp.vars, 'UNK_TK')
  assert isinstance(lmp.vars.UNK_TK, str)
  assert hasattr(lmp.vars, 'UNK_TKID')
  assert isinstance(lmp.vars.UNK_TKID, int)
