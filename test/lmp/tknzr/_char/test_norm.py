r"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.norm`.
"""

from typing import Dict

import pytest

from lmp.tknzr._char import CharTknzr


def test_nfkc(char_tknzr: CharTknzr, non_nfkc_txt: Dict[str, str]):
    r"""Test output text is normalized with NFKC."""

    assert char_tknzr.norm(non_nfkc_txt['input']) == non_nfkc_txt['output']


def test_collapse_whitespace(char_tknzr: CharTknzr, cws_txt: Dict[str, str]):
    r"""Test output text collapse whitespaces."""

    assert char_tknzr.norm(cws_txt['input']) == cws_txt['output']


def test_strip_whitespace(char_tknzr: CharTknzr, htws_txt: Dict[str, str]):
    r"""Test output text is stripped."""

    assert char_tknzr.norm(htws_txt['input']) == htws_txt['output']


@pytest.mark.parametrize(
    "test_cased",
    [
        (
            True
        ),
        (
            False
        ),
    ]
)
def test_lower_case(test_cased: bool, case_txt: Dict[str, str]):
    r"""Test output text is convert to lower case."""

    tknzr = CharTknzr(
        is_uncased=test_cased,
        max_vocab=-1,
        min_count=1,
    )

    if tknzr.is_uncased:
        assert tknzr.norm(case_txt['input']) == case_txt['output']
    else:
        assert tknzr.norm(case_txt['input']) == case_txt['input']
