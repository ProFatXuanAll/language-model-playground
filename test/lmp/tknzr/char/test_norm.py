"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.norm`.
"""

from typing import Dict

from lmp.tknzr import CharTknzr


def test_nfkc(
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  nfkc_txt: Dict[str, str],
  tk2id: Dict[str, int],
) -> None:
  """Normalize text with NFKC."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  assert tknzr.norm(nfkc_txt['input']) == nfkc_txt['output']


def test_collapse_whitespace(
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  tk2id: Dict[str, int],
  ws_collapse_txt: Dict[str, str],
) -> None:
  """Collapse consecutive whitespaces."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  assert tknzr.norm(ws_collapse_txt['input']) == ws_collapse_txt['output']


def test_strip_whitespace(
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  tk2id: Dict[str, int],
  ws_strip_txt: Dict[str, str],
) -> None:
  """Strip text."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  assert tknzr.norm(ws_strip_txt['input']) == ws_strip_txt['output']


def test_uncased(
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  tk2id: Dict[str, int],
  uncased_txt: Dict[str, str],
) -> None:
  """Convert output text to lower cases when ``is_uncased == True``."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  assert (is_uncased and tknzr.norm(uncased_txt['input']) == uncased_txt['output']) or \
    (not is_uncased and tknzr.norm(uncased_txt['input']) == uncased_txt['input'])
