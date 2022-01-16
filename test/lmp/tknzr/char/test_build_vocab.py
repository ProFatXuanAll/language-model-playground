"""Test the construction of character tokenizer's vocabulary.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.build_vocab`.
- :py:meth:`lmp.tknzr.CharTknzr.vocab_size`.
"""

import unicodedata

from lmp.tknzr import CharTknzr


def test_empty_build() -> None:
  """Build nothing when given empty list."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab([])
  assert tknzr.vocab_size == 4
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
  }


def test_no_limit_build() -> None:
  """Include all tokens when ``max_vocab == -1``."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  CJK_txt = [chr(i) for i in range(ord('\u4e00'), ord('\u9fff') + 1)]
  norm_CJK_txt = [tknzr.norm(t) for t in CJK_txt]
  tknzr.build_vocab(CJK_txt)
  assert tknzr.vocab_size == len(set(norm_CJK_txt)) + 4
  assert all(map(lambda tk: tk in tknzr.tk2id, norm_CJK_txt))


def test_limit_build() -> None:
  """Must have correct vocabulary size."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=0)
  tknzr.build_vocab([chr(i) for i in range(65536)])
  assert tknzr.vocab_size == 10


def test_sort_by_occurrence_counts() -> None:
  """Sort vocabulary by occurrence counts."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['c', 'bc', 'abc'])
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'c': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'b': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 3,
  }


def test_minimum_occurrence_counts() -> None:
  """Must satisfy minumum occurrence counts to include tokens in vocabulary."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=2)
  tknzr.build_vocab(['c', 'bc', 'abc'])
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'c': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'b': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
  }


def test_normalization() -> None:
  """Must normalize text first."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['０', '０é'])
  assert tknzr.tk2id == {
    CharTknzr.bos_tk:
      CharTknzr.bos_tkid,
    CharTknzr.eos_tk:
      CharTknzr.eos_tkid,
    CharTknzr.pad_tk:
      CharTknzr.pad_tkid,
    CharTknzr.unk_tk:
      CharTknzr.unk_tkid,
    unicodedata.normalize('NFKC', '０'):
      max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    unicodedata.normalize('NFKC', 'é'):
      max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
  }


def test_case_sensitive() -> None:
  """Must be case-sensitive when ``is_uncased = False``."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a', 'A'])
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'A': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
  }


def test_case_insensitive() -> None:
  """Must be case-insensitive when ``is_uncased = True``."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a', 'A'])
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
  }


def test_continue_build() -> None:
  """Build vocabulary based on existed vocabulary."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['a'])
  tknzr.build_vocab(['b'])
  tknzr.build_vocab(['c'])
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'b': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
    'c': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 3,
  }
