"""Setup fixtures for testing :py:mod:`lmp.util.cfg`."""

import os
from typing import Callable

import pytest

import lmp


@pytest.fixture
def cfg_file_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Mock configuration save file path.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.cfg.FILE_NAME)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_file_path
