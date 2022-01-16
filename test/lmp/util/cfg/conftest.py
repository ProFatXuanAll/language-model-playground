"""Setup fixtures for testing :py:mod:`lmp.util.cfg`."""

import os

import pytest

import lmp


@pytest.fixture
def clean_cfg(
  exp_name: str,
  request,
):
  """Clean up saved configuration file."""
  file_dir = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  file_path = os.path.join(file_dir, lmp.util.cfg.CFG_NAME)

  def remove():
    if os.path.exists(file_path):
      os.remove(file_path)
    if os.path.exists(file_dir) and not os.listdir(file_dir):
      os.removedirs(file_dir)

  request.addfinalizer(remove)
