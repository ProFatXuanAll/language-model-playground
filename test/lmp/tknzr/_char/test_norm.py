r"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.norm`.
"""

from typing import Dict

import pytest

from lmp.tknzr import CharTknzr


def test_nfkc(char_tknzr: CharTknzr, non_nfkc_txt: Dict[str, str]):
    r"""Normalize output text with NFKC."""

    assert char_tknzr.norm(non_nfkc_txt['input']) == non_nfkc_txt['output']


def test_collapse_whitespace(char_tknzr: CharTknzr, cws_txt: Dict[str, str]):
    r"""Collapse whitespaces in output text."""

    assert char_tknzr.norm(cws_txt['input']) == cws_txt['output']


def test_strip_whitespace(char_tknzr: CharTknzr, htws_txt: Dict[str, str]):
    r"""Strip output text."""

    assert char_tknzr.norm(htws_txt['input']) == htws_txt['output']


@pytest.mark.parametrize(
    'is_uncased',
    [
        True,
        False,
    ]
)
def test_lower_case(is_uncased: bool, cased_txt: Dict[str, str]):
    r"""Convert output text to lowercase when ``is_uncased == True``."""

    tknzr = CharTknzr(
        is_uncased=is_uncased,
        max_vocab=-1,
        min_count=1,
        tk2id=None,
    )

    if tknzr.is_uncased:
        assert tknzr.norm(cased_txt['input']) == cased_txt['output']
    else:
        assert tknzr.norm(cased_txt['input']) == cased_txt['input']
