"""Test tokenization and detokenization.

Test target:
- :py:meth:`lmp.tknzr._char.CharTknzr.dtknz`.
- :py:meth:`lmp.tknzr._char.CharTknzr.tknz`.
"""

import unicodedata

import pytest

from lmp.tknzr._char import CharTknzr
from lmp.vars import BOS_TK, EOS_TK, PAD_TK, UNK_TK


@pytest.fixture
def tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> CharTknzr:
  """Character tokenizer shared in this module."""
  return CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)


def test_tknz(tknzr: CharTknzr) -> None:
  """Tokenize text into characters."""
  # Return empty list when input empty string.
  assert tknzr.tknz('') == []
  # Normalize with NFKC.
  assert tknzr.tknz('０é') == [unicodedata.normalize('NFKC', '０'), unicodedata.normalize('NFKC', 'é')]
  # Case-sensitive and case-insensitive.
  assert (tknzr.is_uncased and tknzr.tknz('ABc') == ['a', 'b', 'c']) or \
    (not tknzr.is_uncased and tknzr.tknz('ABc') == ['A', 'B', 'c'])
  # Collapse consecutive whitespaces.
  assert tknzr.tknz('a  b   c') == ['a', ' ', 'b', ' ', 'c']
  # Strip whitespaces.
  assert tknzr.tknz('  abc  ') == ['a', 'b', 'c']
  # Avoid tokenizing special tokens.
  assert tknzr.tknz(f'{BOS_TK}a{UNK_TK}bc{EOS_TK}{PAD_TK}{PAD_TK}') == [
    BOS_TK,
    'a',
    UNK_TK,
    'b',
    'c',
    EOS_TK,
    PAD_TK,
    PAD_TK,
  ]


def test_dtknz(tknzr: CharTknzr) -> None:
  """Detokenize characters back to text."""
  # Return empty string when input empty list.
  assert tknzr.dtknz([]) == ''
  # Normalize with NFKC.
  assert tknzr.dtknz(['０', 'é']) == unicodedata.normalize('NFKC', '０é')
  # Case-sensitive and case-insensitive.
  assert (tknzr.is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'abc') or \
    (not tknzr.is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'ABc')
  # Collapse consecutive whitespaces.
  assert tknzr.dtknz(['a', ' ', ' ', 'b', ' ', ' ', ' ', 'c']) == 'a b c'
  # Strip whitespaces.
  assert tknzr.dtknz([' ', 'a', 'b', 'c', ' ']) == 'abc'
  # Correct joint special tokens.
  assert tknzr.dtknz(
    [
      BOS_TK,
      'a',
      UNK_TK,
      'b',
      'c',
      EOS_TK,
      PAD_TK,
      PAD_TK,
    ]
  ) == f'{BOS_TK}a{UNK_TK}bc{EOS_TK}{PAD_TK}{PAD_TK}'
