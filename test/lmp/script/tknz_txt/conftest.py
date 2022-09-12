"""Setup fixtures for testing :py:mod:`lmp.script.tknz_txt`."""

import pytest

import lmp.util.cfg
import lmp.util.tknzr
import lmp.vars
from lmp.tknzr import BPETknzr, CharTknzr, WsTknzr


@pytest.fixture
def bpe_tknzr(exp_name: str, request, tknzr_file_path: None) -> CharTknzr:
  """BPE tokenizer example."""
  tknzr = BPETknzr()
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr


@pytest.fixture
def char_tknzr(exp_name: str, request, tknzr_file_path: None) -> CharTknzr:
  """Character tokenizer example."""
  tknzr = CharTknzr()
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr


@pytest.fixture
def ws_tknzr(exp_name: str, request, tknzr_file_path: None) -> WsTknzr:
  """Whitespace tokenizer example."""
  tknzr = WsTknzr()
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr
