"""Setup fixtures for testing :py:mod:`lmp.script.tknz_txt`."""

import pytest

import lmp.util.cfg
import lmp.util.path
import lmp.util.tknzr
from lmp.tknzr import CharTknzr, WsTknzr


@pytest.fixture
def char_tknzr(exp_name: str, request, tknzr_file_path: None) -> CharTknzr:
  """Character tokenizer example."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr


@pytest.fixture
def ws_tknzr(exp_name: str, request, tknzr_file_path: None) -> WsTknzr:
  """Whitespace tokenizer example."""
  tknzr = WsTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  return tknzr
