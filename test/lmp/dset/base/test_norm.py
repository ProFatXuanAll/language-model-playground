"""Test text normalization.

Test target:
- :py:meth:`lmp.dset.BaseDset.norm`.
"""

from typing import Dict

from lmp.dset import BaseDset


def test_nfkc(nfkc_txt: Dict[str, str]) -> None:
  """Normalize text with NFKC."""
  assert BaseDset.norm(nfkc_txt['input']) == nfkc_txt['output']


def test_collapse_whitespace(ws_collapse_txt: Dict[str, str]) -> None:
  """Collapse consecutive whitespaces."""
  assert BaseDset.norm(ws_collapse_txt['input']) == ws_collapse_txt['output']


def test_strip_whitespace(ws_strip_txt: Dict[str, str]) -> None:
  """Strip text."""
  assert BaseDset.norm(ws_strip_txt['input']) == ws_strip_txt['output']
