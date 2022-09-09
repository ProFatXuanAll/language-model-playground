"""Test construction utilities for all tokenizers.

Test target:
- :py:meth:`lmp.util.tknzr.create`.
"""

import lmp.util.tknzr
from lmp.tknzr import BPETknzr, CharTknzr, WsTknzr


def test_create_bpe_tknzr(is_uncased: bool, max_vocab: int, min_count: int, n_merge: int) -> None:
  """Test construction for BPE tokenizer."""
  tknzr = lmp.util.tknzr.create(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
    n_merge=n_merge,
    tknzr_name=BPETknzr.tknzr_name,
  )
  assert isinstance(tknzr, BPETknzr)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.n_merge == n_merge

def test_create_char_tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Test construction for character tokenizer."""
  tknzr = lmp.util.tknzr.create(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
    tknzr_name=CharTknzr.tknzr_name,
  )
  assert isinstance(tknzr, CharTknzr)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count


def test_create_ws_tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Test construction for whitespace tokenizer."""
  tknzr = lmp.util.tknzr.create(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
    tknzr_name=WsTknzr.tknzr_name,
  )
  assert isinstance(tknzr, WsTknzr)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
