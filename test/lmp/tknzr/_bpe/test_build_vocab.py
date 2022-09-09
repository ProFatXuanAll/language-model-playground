"""Test the construction of character tokenizer's vocabulary.

Test target:
- :py:meth:`lmp.tknzr._bpe.BPETknzr.build_vocab`.
- :py:meth:`lmp.tknzr._bpe.BPETknzr.vocab_size`.
"""

import unicodedata

from lmp.tknzr._bpe import BPETknzr, EOW_TK
from lmp.vars import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID


def test_empty_build() -> None:
  """Build nothing when given empty list."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=-1, min_count=0, n_merge=10)
  tknzr.build_vocab([])
  assert tknzr.vocab_size == 4
  assert tknzr.tk2id == {BOS_TK: BOS_TKID, EOS_TK: EOS_TKID, PAD_TK: PAD_TKID, UNK_TK: UNK_TKID}


def test_no_limit_build() -> None:
  """Include all tokens when ``max_vocab == -1``."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=-1, min_count=0, n_merge=10)
  CJK_txt = [chr(i) for i in range(ord('\u4e00'), ord('\u9fff') + 1)]
  norm_CJK_txt = [tknzr.norm(t) for t in CJK_txt]
  tknzr.build_vocab(CJK_txt)
  assert tknzr.vocab_size == len(set(norm_CJK_txt)) + 4
  assert all(map(lambda tk: tk in tknzr.tk2id or f'{tk}{EOW_TK}', norm_CJK_txt))


def test_limit_build() -> None:
  """Must have correct vocabulary size."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=10, min_count=0, n_merge=10)
  tknzr.build_vocab([chr(i) for i in range(65536)])
  assert tknzr.vocab_size == 10


def test_sort_by_occurrence_counts() -> None:
  """Sort vocabulary by occurrence counts."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=-1, min_count=0, n_merge=10)
  tknzr.build_vocab(['c', 'bc', 'abc'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    f'c{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    'b': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
    'a': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 3,
    f'bc{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 4,
    f'abc{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 5,
  }


def test_minimum_occurrence_counts() -> None:
  """Must satisfy minumum occurrence counts to include tokens in vocabulary."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=-1, min_count=2, n_merge=10)
  tknzr.build_vocab(['c', 'bc', 'abc'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    f'c{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    'b': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
    f'bc{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 3,
  }


def test_normalization() -> None:
  """Must normalize text first."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['０', 'é'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    f"{unicodedata.normalize('NFKC', '０')}{EOW_TK}": max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    f"{unicodedata.normalize('NFKC', 'é')}{EOW_TK}": max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
  }


def test_case_sensitive() -> None:
  """Must be case-sensitive when ``is_uncased = False``."""
  tknzr = BPETknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a', 'A'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    f'a{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    f'A{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
  }


def test_case_insensitive() -> None:
  """Must be case-insensitive when ``is_uncased = True``."""
  tknzr = BPETknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a', 'A'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    f'a{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
  }


def test_continue_build() -> None:
  """Build vocabulary based on existed vocabulary."""
  tknzr = BPETknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a'])
  tknzr.build_vocab(['b'])
  tknzr.build_vocab(['c'])
  assert tknzr.tk2id == {
    BOS_TK: BOS_TKID,
    EOS_TK: EOS_TKID,
    PAD_TK: PAD_TKID,
    UNK_TK: UNK_TKID,
    f'a{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 1,
    f'b{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 2,
    f'c{EOW_TK}': max(BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID) + 3,
  }
