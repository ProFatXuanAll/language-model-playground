"""Setup fixtures for testing :py:mod:`lmp.tknzr`."""

import os
from typing import Dict

import pytest

import lmp.util.path
from lmp.tknzr import BaseTknzr


@pytest.fixture(params=[False, True])
def is_uncased(request) -> bool:
  """Respect cases if set to ``False``."""
  return request.param


@pytest.fixture(params=[-1, 10000])
def max_vocab(request) -> int:
  """Maximum vocabulary size."""
  return request.param


@pytest.fixture(params=[0, 10])
def min_count(request) -> int:
  """Minimum token occurrence counts."""
  return request.param


@pytest.fixture(params=[
  {
    'input': 'ABC',
    'output': 'abc'
  },
  {
    'input': 'abc',
    'output': 'abc'
  },
])
def uncased_txt(request) -> Dict[str, str]:
  """Case-insensitive text."""
  return request.param


@pytest.fixture
def tknzr_file_path(request, exp_name: str) -> str:
  """Tokenizer save file path.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, BaseTknzr.file_name)

  def fin() -> None:
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_file_path
