"""Setup fixtures for testing :py:mod:`lmp.script.train_tknzr`."""

import os

import pytest

import lmp.util.path
from lmp.tknzr import BaseTknzr


@pytest.fixture(params=[False, True])
def is_uncased(request) -> bool:
  """Respect cases if set to ``False``."""
  return request.param


@pytest.fixture(params=[-1, 100])
def max_vocab(request) -> int:
  """Maximum vocabulary size."""
  return request.param


@pytest.fixture(params=[0, 10])
def min_count(request) -> int:
  """Minimum token occurrence counts."""
  return request.param


@pytest.fixture
def tknzr_file_path(exp_name: str, request) -> str:
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


@pytest.fixture
def cfg_file_path(exp_name: str, request) -> str:
  """Clean up saved configuration file."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.cfg.CFG_NAME)

  def fin():
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_file_path
