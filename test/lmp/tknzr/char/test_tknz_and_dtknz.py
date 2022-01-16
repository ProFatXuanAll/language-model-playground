"""Test tokenization and detokenization.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.dtknz`.
- :py:meth:`lmp.tknzr.CharTknzr.tknz`.
"""

import unicodedata
from typing import Dict

from lmp.tknzr import CharTknzr


def test_tknz(is_uncased: bool, max_vocab: int, min_count: int, tk2id: Dict[str, int]) -> None:
  """Tokenize text into characters."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  # Return empty list when input empty string.
  assert tknzr.tknz('') == []
  # Normalize with NFKC.
  assert tknzr.tknz('０é') == [unicodedata.normalize('NFKC', '０'), unicodedata.normalize('NFKC', 'é')]
  # Case-sensitive and case-insensitive.
  assert (is_uncased and tknzr.tknz('ABc') == ['a', 'b', 'c']) or \
    (not is_uncased and tknzr.tknz('ABc') == ['A', 'B', 'c'])
  # Collapse consecutive whitespaces.
  assert tknzr.tknz('a  b   c') == ['a', ' ', 'b', ' ', 'c']
  # Strip whitespaces.
  assert tknzr.tknz('  abc  ') == ['a', 'b', 'c']
  # Avoid tokenizing special tokens.
  assert (
    tknzr.tknz(f'{CharTknzr.bos_tk}a{CharTknzr.unk_tk}bc{CharTknzr.eos_tk}{CharTknzr.pad_tk}{CharTknzr.pad_tk}') == [
      CharTknzr.bos_tk,
      'a',
      CharTknzr.unk_tk,
      'b',
      'c',
      CharTknzr.eos_tk,
      CharTknzr.pad_tk,
      CharTknzr.pad_tk,
    ]
  )


def test_dtknz(is_uncased: bool, max_vocab: int, min_count: int, tk2id: Dict[str, int]) -> None:
  """Detokenize characters back to text."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  # Return empty string when input empty list.
  assert tknzr.dtknz([]) == ''
  # Normalize with NFKC.
  assert tknzr.dtknz(['０', 'é']) == unicodedata.normalize('NFKC', '０é')
  # Case-sensitive and case-insensitive.
  assert (is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'abc') or \
    (not is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'ABc')
  # Collapse consecutive whitespaces.
  assert tknzr.dtknz(['a', ' ', ' ', 'b', ' ', ' ', ' ', 'c']) == 'a b c'
  # Strip whitespaces.
  assert tknzr.dtknz([' ', 'a', 'b', 'c', ' ']) == 'abc'
  # Correct joint special tokens.
  assert tknzr.dtknz(
    [
      CharTknzr.bos_tk,
      'a',
      CharTknzr.unk_tk,
      'b',
      'c',
      CharTknzr.eos_tk,
      CharTknzr.pad_tk,
      CharTknzr.pad_tk,
    ]
  ) == f'{CharTknzr.bos_tk}a{CharTknzr.unk_tk}bc{CharTknzr.eos_tk}{CharTknzr.pad_tk}{CharTknzr.pad_tk}'
