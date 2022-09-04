"""Setup fixtures for testing :py:mod:`lmp.util.model`."""

import os
from typing import Callable

import pytest

import lmp.vars


@pytest.fixture
def ckpt_dir_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Model checkpoints save path.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.vars.EXP_PATH, exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_dir_path
