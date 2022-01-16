"""Test tokenization and detokenization.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.dtknz`.
- :py:meth:`lmp.tknzr.WsTknzr.tknz`.
"""

import unicodedata
from typing import Dict

from lmp.tknzr import WsTknzr


def test_tknz(is_uncased: bool, max_vocab: int, min_count: int, tk2id: Dict[str, int]) -> None:
  """Tokenize text into characters."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  # Return empty list when input empty string.
  assert tknzr.tknz('') == []
  # Normalize with NFKC.
  assert tknzr.tknz('０ é') == [unicodedata.normalize('NFKC', '０'), unicodedata.normalize('NFKC', 'é')]
  # Case-sensitive and case-insensitive.
  assert (is_uncased and tknzr.tknz('A B c') == ['a', 'b', 'c']) or \
    (not is_uncased and tknzr.tknz('A B c') == ['A', 'B', 'c'])
  # Collapse consecutive whitespaces.
  assert tknzr.tknz('a  b   c') == ['a', 'b', 'c']
  # Strip whitespaces.
  assert tknzr.tknz('  a b c  ') == ['a', 'b', 'c']
  # Avoid tokenizing special tokens.
  assert (
    tknzr.tknz(f'{WsTknzr.bos_tk} a {WsTknzr.unk_tk} b c {WsTknzr.eos_tk} {WsTknzr.pad_tk} {WsTknzr.pad_tk}') == [
      WsTknzr.bos_tk,
      'a',
      WsTknzr.unk_tk,
      'b',
      'c',
      WsTknzr.eos_tk,
      WsTknzr.pad_tk,
      WsTknzr.pad_tk,
    ]
  )


def test_dtknz(is_uncased: bool, max_vocab: int, min_count: int, tk2id: Dict[str, int]) -> None:
  """Detokenize characters back to text."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  # Return empty string when input empty list.
  assert tknzr.dtknz([]) == ''
  # Normalize with NFKC.
  assert tknzr.dtknz(['０', 'é']) == unicodedata.normalize('NFKC', '０ é')
  # Case-sensitive and case-insensitive.
  assert (is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'a b c') or \
    (not is_uncased and tknzr.dtknz(['A', 'B', 'c']) == 'A B c')
  # Collapse consecutive whitespaces.
  assert tknzr.dtknz(['a', ' ', ' ', 'b', ' ', ' ', ' ', 'c']) == 'a b c'
  # Strip whitespaces.
  assert tknzr.dtknz([' ', 'a', 'b', 'c', ' ']) == 'a b c'
  # Correct joint special tokens.
  assert tknzr.dtknz(
    [
      WsTknzr.bos_tk,
      'a',
      WsTknzr.unk_tk,
      'b',
      'c',
      WsTknzr.eos_tk,
      WsTknzr.pad_tk,
      WsTknzr.pad_tk,
    ]
  ) == f'{WsTknzr.bos_tk} a {WsTknzr.unk_tk} b c {WsTknzr.eos_tk} {WsTknzr.pad_tk} {WsTknzr.pad_tk}'
