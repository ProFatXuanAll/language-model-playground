"""Test loading utilities for all tokenizers.

Test target:
- :py:meth:`lmp.util.tknzr.load`.
"""

import lmp.util.tknzr
from lmp.tknzr import CharTknzr, WsTknzr


def test_load_char_tknzr(
  exp_name: str,
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  tknzr_file_path: str,
) -> None:
  """Ensure consistency between save and load."""
  tk2id = {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'b': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
    'c': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 3,
  }
  tknzr = lmp.util.tknzr.create(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
    tk2id=tk2id,
    tknzr_name=CharTknzr.tknzr_name,
  )
  tknzr.save(exp_name=exp_name)
  load_tknzr = lmp.util.tknzr.load(exp_name=exp_name, tknzr_name=CharTknzr.tknzr_name)
  assert isinstance(load_tknzr, CharTknzr)
  assert load_tknzr.is_uncased == tknzr.is_uncased
  assert load_tknzr.max_vocab == tknzr.max_vocab
  assert load_tknzr.min_count == tknzr.min_count
  assert load_tknzr.tk2id == tknzr.tk2id
  assert load_tknzr.id2tk == tknzr.id2tk


def test_load_ws_tknzr(
  exp_name: str,
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  tknzr_file_path: str,
) -> None:
  """Ensure consistency between save and load."""
  tk2id = {
    WsTknzr.bos_tk: WsTknzr.bos_tkid,
    WsTknzr.eos_tk: WsTknzr.eos_tkid,
    WsTknzr.pad_tk: WsTknzr.pad_tkid,
    WsTknzr.unk_tk: WsTknzr.unk_tkid,
    'a': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 1,
    'b': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 2,
    'c': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 3,
  }
  tknzr = lmp.util.tknzr.create(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
    tk2id=tk2id,
    tknzr_name=WsTknzr.tknzr_name,
  )
  tknzr.save(exp_name=exp_name)
  load_tknzr = lmp.util.tknzr.load(exp_name=exp_name, tknzr_name=WsTknzr.tknzr_name)
  assert isinstance(load_tknzr, WsTknzr)
  assert load_tknzr.is_uncased == tknzr.is_uncased
  assert load_tknzr.max_vocab == tknzr.max_vocab
  assert load_tknzr.min_count == tknzr.min_count
  assert load_tknzr.tk2id == tknzr.tk2id
  assert load_tknzr.id2tk == tknzr.id2tk
