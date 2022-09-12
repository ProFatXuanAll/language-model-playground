"""Test tokenization and detokenization.

Test target:
- :py:meth:`lmp.tknzr._bpe.BPETknzr.dtknz`.
- :py:meth:`lmp.tknzr._bpe.BPETknzr.tknz`.
"""

import unicodedata

import pytest

from lmp.tknzr._bpe import EOW_TK, BPETknzr
from lmp.vars import BOS_TK, EOS_TK, PAD_TK, UNK_TK


@pytest.fixture
def tknzr(is_uncased: bool, max_vocab: int, min_count: int, n_merge: int) -> BPETknzr:
  """Character tokenizer shared in this module."""
  return BPETknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, n_merge=n_merge)


def test_tknz(tknzr: BPETknzr) -> None:
  """Tokenize text into characters."""
  # Return empty list when input empty string.
  assert tknzr.tknz('') == []
  # Normalize with NFKC.
  assert tknzr.tknz('０é') == [f"{unicodedata.normalize('NFKC', '０é')}{EOW_TK}"]
  # Case-sensitive and case-insensitive.
  assert (tknzr.is_uncased and tknzr.tknz('ABc') == [f'abc{EOW_TK}']) or \
    (not tknzr.is_uncased and tknzr.tknz('ABc') == [f'ABc{EOW_TK}'])
  # Collapse consecutive whitespaces.
  assert tknzr.tknz('a  b   c') == [f'a{EOW_TK}', f'b{EOW_TK}', f'c{EOW_TK}']
  # Strip whitespaces.
  assert tknzr.tknz('  abc  ') == [f'abc{EOW_TK}']
  # Avoid tokenizing special tokens.
  assert tknzr.tknz(f'{BOS_TK}a{UNK_TK}bc{EOS_TK}{PAD_TK}{PAD_TK}') == [
    BOS_TK,
    f'a{EOW_TK}',
    UNK_TK,
    f'bc{EOW_TK}',
    EOS_TK,
    PAD_TK,
    PAD_TK,
  ]


def test_dtknz(tknzr: BPETknzr) -> None:
  """Detokenize characters back to text."""
  # Return empty string when input empty list.
  assert tknzr.dtknz([]) == ''
  # Normalize with NFKC.
  assert tknzr.dtknz([f'０é{EOW_TK}']) == unicodedata.normalize('NFKC', '０é')
  # Case-sensitive and case-insensitive.
  assert (tknzr.is_uncased and tknzr.dtknz(['A', 'B', f'c{EOW_TK}']) == 'abc') or \
    (not tknzr.is_uncased and tknzr.dtknz(['A', 'B', f'c{EOW_TK}']) == 'ABc')
  # Collapse consecutive whitespaces.
  assert tknzr.dtknz(['a', ' ', ' ', 'b', ' ', ' ', ' ', f'c{EOW_TK}']) == 'abc'
  # Strip whitespaces.
  assert tknzr.dtknz([' ', 'a', 'b', f'c{EOW_TK}', ' ']) == 'abc'
  # Correct joint special tokens.
  assert tknzr.dtknz(
    [
      BOS_TK,
      f'a{EOW_TK}',
      UNK_TK,
      'b',
      f'c{EOW_TK}',
      EOS_TK,
      PAD_TK,
      PAD_TK,
    ]
  ) == f'{BOS_TK} a {UNK_TK} bc {EOS_TK} {PAD_TK} {PAD_TK}'
