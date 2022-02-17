"""Test the construction of whitespace tokenizer's vocabulary.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.build_vocab`.
- :py:meth:`lmp.tknzr._ws.WsTknzr.vocab_size`.
"""

import unicodedata

from lmp.tknzr._base import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID
from lmp.tknzr._ws import WsTknzr


def test_empty_build() -> None:
  """Build nothing when given empty list."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab([])
  assert tknzr.vocab_size == 4
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
  }


def test_no_limit_build() -> None:
  """Include all tokens when ``max_vocab == -1``."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  CJK_txt = [chr(i) for i in range(ord('\u4e00'), ord('\u9fff') + 1)]
  norm_CJK_txt = [tknzr.norm(t) for t in CJK_txt]
  tknzr.build_vocab(CJK_txt)
  assert tknzr.vocab_size == len(set(norm_CJK_txt)) + 4
  assert all(map(lambda tk: tk in tknzr.tk2id, norm_CJK_txt))


def test_limit_build() -> None:
  """Must have correct vocabulary size."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=0)
  tknzr.build_vocab([chr(i) for i in range(65536)])
  assert tknzr.vocab_size == 10


def test_sort_by_occurrence_counts() -> None:
  """Sort vocabulary by occurrence counts."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['c', 'b c', 'a b c'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    'c': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    'b': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
    'a': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 3,
  }


def test_minimum_occurrence_counts() -> None:
  """Must satisfy minumum occurrence counts to include tokens in vocabulary."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=2)
  tknzr.build_vocab(['c', 'b c', 'a b c'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    'c': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    'b': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
  }


def test_normalization() -> None:
  """Must normalize text first."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['０', '０ é'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    unicodedata.normalize('NFKC', '０'): max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    unicodedata.normalize('NFKC', 'é'): max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
  }


def test_case_sensitive() -> None:
  """Must be case-sensitive when ``is_uncased = False``."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a', 'A'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    'a': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    'A': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
  }


def test_case_insensitive() -> None:
  """Must be case-insensitive when ``is_uncased = True``."""
  tknzr = WsTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a', 'A'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    'a': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
  }


def test_continue_build() -> None:
  """Build vocabulary based on existed vocabulary."""
  tknzr = WsTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a'])
  tknzr.build_vocab(['b'])
  tknzr.build_vocab(['c'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    'a': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    'b': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
    'c': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 3,
  }
