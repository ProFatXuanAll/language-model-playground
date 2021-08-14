r"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.norm`.
"""

from typing import Dict

import pytest

from lmp.tknzr._ws import WsTknzr


def test_nfkc(ws_tknzr: WsTknzr, non_nfkc_txt: Dict[str, str]):
    r"""Test output text is normalized with NFKC."""

    assert ws_tknzr.norm(non_nfkc_txt['input']) == non_nfkc_txt['output']


def test_collapse_whitespace(ws_tknzr: WsTknzr, cws_txt: Dict[str, str]):
    r"""Test output text collapse whitespaces."""

    assert ws_tknzr.norm(cws_txt['input']) == cws_txt['output']


def test_strip_whitespace(ws_tknzr: WsTknzr, htws_txt: Dict[str, str]):
    r"""Test output text is stripped."""

    assert ws_tknzr.norm(htws_txt['input']) == htws_txt['output']


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
def test_lower_case(test_cased, case_txt: Dict[str, str]):
    r"""Test output text is convert to lower case."""

    tknzr = WsTknzr(
        is_uncased=test_cased,
        max_vocab=-1,
        min_count=1,
    )

    if tknzr.is_uncased:
        assert tknzr.norm(case_txt['input']) == case_txt['output']
    else:
        assert tknzr.norm(case_txt['input']) == case_txt['input']
