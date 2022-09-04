r"""Setup fixtures for testing :py:mod:`lmp.util.log`."""

import os
from typing import Callable

import pytest

import lmp.vars


@pytest.fixture
def log_dir_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  r"""Clean up tensorboard loggings."""
  abs_dir_path = os.path.join(lmp.vars.LOG_PATH, exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_dir_path
