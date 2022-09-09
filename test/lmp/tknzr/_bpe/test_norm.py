"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr._bpe.BPETknzr.norm`.
"""

from typing import Dict

import pytest

from lmp.tknzr._bpe import BPETknzr


@pytest.fixture
def tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> BPETknzr:
  """Character tokenizer shared in this module."""
  return BPETknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)


def test_nfkc(nfkc_txt: Dict[str, str], tknzr: BPETknzr) -> None:
  """Normalize text with NFKC."""
  assert tknzr.norm(nfkc_txt['input']) == nfkc_txt['output']


def test_collapse_whitespace(tknzr: BPETknzr, ws_collapse_txt: Dict[str, str]) -> None:
  """Collapse consecutive whitespaces."""
  assert tknzr.norm(ws_collapse_txt['input']) == ws_collapse_txt['output']


def test_strip_whitespace(tknzr: BPETknzr, ws_strip_txt: Dict[str, str]) -> None:
  """Strip text."""
  assert tknzr.norm(ws_strip_txt['input']) == ws_strip_txt['output']


def test_uncased(tknzr: BPETknzr, uncased_txt: Dict[str, str]) -> None:
  """Convert output text to lower cases when ``is_uncased == True``."""
  assert (tknzr.is_uncased and tknzr.norm(uncased_txt['input']) == uncased_txt['output']) or \
    (not tknzr.is_uncased and tknzr.norm(uncased_txt['input']) == uncased_txt['input'])
