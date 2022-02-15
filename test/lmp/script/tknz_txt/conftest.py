"""Setup fixtures for testing :py:mod:`lmp.script.tknz_txt`."""

import os

import pytest

import lmp.util.cfg
import lmp.util.path
import lmp.util.tknzr
from lmp.tknzr import CharTknzr, WsTknzr


@pytest.fixture
def seed() -> int:
  """Random seed."""
  return 42


@pytest.fixture
def tknzr_file_path(exp_name: str, request) -> None:
  """Cleanup saved tokenizer."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.tknzr.FILE_NAME)

  def fin() -> None:
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)


@pytest.fixture
def char_tknzr(exp_name: str, request, tknzr_file_path: None) -> CharTknzr:
  """Character tokenizer example."""
  tknzr = CharTknzr(is_uncased=True, max_seq_len=128, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr


@pytest.fixture
def ws_tknzr(exp_name: str, request, tknzr_file_path: None) -> WsTknzr:
  """Whitespace tokenizer example."""
  tknzr = WsTknzr(is_uncased=True, max_seq_len=128, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr
