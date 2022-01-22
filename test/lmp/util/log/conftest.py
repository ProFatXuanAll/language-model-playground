r"""Setup fixtures for testing :py:mod:`lmp.util.log`."""

import os

import pytest

import lmp.util.path


@pytest.fixture
def log_dir_path(exp_name: str, request) -> str:
  r"""Clean up tensorboard loggings."""
  abs_dir_path = os.path.join(lmp.util.path.LOG_PATH, exp_name)

  def fin() -> None:
    for log_file in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, log_file))
    os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_dir_path
