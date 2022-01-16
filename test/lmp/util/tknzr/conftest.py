"""Setup fixtures for testing :py:mod:`lmp.util.tknzr`."""

import os

import pytest

import lmp


@pytest.fixture
def clean_tknzr(exp_name: str, request):
  """Clean up saving tokenizers."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def remove():
    for file_name in os.listdir(abs_dir_path):
      abs_file_path = os.path.join(abs_dir_path, file_name)
      os.remove(abs_file_path)

    os.removedirs(abs_dir_path)

  request.addfinalizer(remove)
