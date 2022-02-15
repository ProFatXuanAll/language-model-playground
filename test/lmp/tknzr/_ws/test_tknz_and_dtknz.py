"""Test tokenization and detokenization.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.dtknz`.
- :py:meth:`lmp.tknzr._ws.WsTknzr.tknz`.
"""

import unicodedata

import pytest

from lmp.tknzr._base import BOS_TK, EOS_TK, PAD_TK, UNK_TK
from lmp.tknzr._ws import WsTknzr


@pytest.fixture
def tknzr(is_uncased: bool, max_seq_len: int, max_vocab: int, min_count: int) -> WsTknzr:
  """Whitespace tokenizer shared in this module."""
  return WsTknzr(is_uncased=is_uncased, max_seq_len=max_seq_len, max_vocab=max_vocab, min_count=min_count)


def test_tknz(tknzr: WsTknzr) -> None:
  """Tokenize text into characters."""
  # Return empty list when input empty string.
  assert tknzr.tknz('') == []
  # Normalize with NFKC.
  assert tknzr.tknz('０ é') == [unicodedata.normalize('NFKC', '０'), unicodedata.normalize('NFKC', 'é')]
  # Case-sensitive and case-insensitive.
  assert (tknzr.is_uncased and tknzr.tknz('A B c') == ['a', 'b', 'c']) or \
    (not tknzr.is_uncased and tknzr.tknz('A B c') == ['A', 'B', 'c'])
  # Collapse consecutive whitespaces.
  assert tknzr.tknz('a  b   c') == ['a', 'b', 'c']
  # Strip whitespaces.
  assert tknzr.tknz('  a b c  ') == ['a', 'b', 'c']
  # Avoid tokenizing special tokens.
  assert tknzr.tknz(f'{BOS_TK} a {UNK_TK} b c {EOS_TK} {PAD_TK} {PAD_TK}') == [
    BOS_TK,
    'a',
    UNK_TK,
    'b',
    'c',
    EOS_TK,
    PAD_TK,
    PAD_TK,
  ]


def test_dtknz(tknzr: WsTknzr) -> None:
  """Detokenize characters back to text."""
  # Return empty string when input empty list.
  assert tknzr.dtknz([]) == ''
  # Normalize with NFKC.
  assert tknzr.dtknz(['０', 'é']) == unicodedata.normalize('NFKC', '０ é')
  # Case-sensitive and case-insensitive.
  assert (tknzr.is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'a b c') or \
    (not tknzr.is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'A B c')
  # Collapse consecutive whitespaces.
  assert tknzr.dtknz(['a', ' ', ' ', 'b', ' ', ' ', ' ', 'c']) == 'a b c'
  # Strip whitespaces.
  assert tknzr.dtknz([' ', 'a', 'b', 'c', ' ']) == 'a b c'
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
  ) == f'{BOS_TK} a {UNK_TK} b c {EOS_TK} {PAD_TK} {PAD_TK}'
