"""Test construction utilities for all tokenizers.

Test target:
- :py:meth:`lmp.util.tknzr.create`.
"""

import lmp.util.tknzr
from lmp.tknzr import CharTknzr, WsTknzr


def test_create_char_tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Test construction for character tokenizer."""
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
  assert isinstance(tknzr, CharTknzr)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == tk2id


def test_create_ws_tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Test construction for whitespace tokenizer."""
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
  assert isinstance(tknzr, WsTknzr)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == tk2id
