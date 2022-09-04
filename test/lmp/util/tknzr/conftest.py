"""Setup fixtures for testing :py:mod:`lmp.util.tknzr`."""

import os
from typing import Callable

import pytest

import lmp.util.tknzr
import lmp.vars


@pytest.fixture
def tknzr_file_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Tokenizer save file path.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.vars.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.tknzr.FILE_NAME)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_file_path
