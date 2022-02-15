"""Test saving and loading utilities for all tokenizers.

Test target:
- :py:meth:`lmp.util.tknzr.load`.
"""

import os

import lmp.util.tknzr
from lmp.tknzr import CharTknzr, WsTknzr


def test_char_tknzr(
  exp_name: str,
  is_uncased: bool,
  max_seq_len: int,
  max_vocab: int,
  min_count: int,
  tknzr_file_path: str,
) -> None:
  """Ensure consistency between save and load."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_seq_len=max_seq_len, max_vocab=max_vocab, min_count=min_count)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  assert os.path.exists(tknzr_file_path)

  load_tknzr = lmp.util.tknzr.load(exp_name=exp_name)
  assert isinstance(load_tknzr, CharTknzr)
  assert load_tknzr.is_uncased == tknzr.is_uncased
  assert load_tknzr.max_seq_len == tknzr.max_seq_len
  assert load_tknzr.max_vocab == tknzr.max_vocab
  assert load_tknzr.min_count == tknzr.min_count
  assert load_tknzr.tk2id == tknzr.tk2id
  assert load_tknzr.id2tk == tknzr.id2tk


def test_ws_tknzr(
  exp_name: str,
  is_uncased: bool,
  max_seq_len: int,
  max_vocab: int,
  min_count: int,
  tknzr_file_path: str,
) -> None:
  """Ensure consistency between save and load."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_seq_len=max_seq_len, max_vocab=max_vocab, min_count=min_count)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  assert os.path.exists(tknzr_file_path)

  load_tknzr = lmp.util.tknzr.load(exp_name=exp_name)
  assert isinstance(load_tknzr, WsTknzr)
  assert load_tknzr.is_uncased == tknzr.is_uncased
  assert load_tknzr.max_seq_len == tknzr.max_seq_len
  assert load_tknzr.max_vocab == tknzr.max_vocab
  assert load_tknzr.min_count == tknzr.min_count
  assert load_tknzr.tk2id == tknzr.tk2id
  assert load_tknzr.id2tk == tknzr.id2tk
