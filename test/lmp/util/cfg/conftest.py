"""Setup fixtures for testing :py:mod:`lmp.util.cfg`."""

import os

import pytest

import lmp


@pytest.fixture
def cfg_file_path(exp_name: str, request) -> str:
  """Clean up saved configuration file."""
  abs_file_dir = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_file_dir, lmp.util.cfg.CFG_NAME)

  def fin():
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)
    if os.path.exists(abs_file_dir) and not os.listdir(abs_file_dir):
      os.removedirs(abs_file_dir)

  request.addfinalizer(fin)
  return abs_file_path
