r"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.norm`.
"""

from typing import Dict

from lmp.tknzr._base import BaseTknzr


def test_nfkc(subclss_tknzr: BaseTknzr, non_nfkc_txt: Dict[str, str]):
    r"""Test output text is normalized with NFKC."""
    assert subclss_tknzr.norm(non_nfkc_txt['input']) == non_nfkc_txt['output']


def test_collapse_whitespace(
        subclss_tknzr: BaseTknzr,
        cws_txt: Dict[str, str]
):
    r"""Test output text collapse whitespaces."""
    assert subclss_tknzr.norm(cws_txt['input']) == cws_txt['output']


def test_strip_whitespace(subclss_tknzr: BaseTknzr, htws_txt: Dict[str, str]):
    r"""Test output text is stripped."""
    assert subclss_tknzr.norm(htws_txt['input']) == htws_txt['output']


def test_lower_case(subclss_tknzr: BaseTknzr, case_txt: Dict[str, str]):
    r"""Test output text is convert to lower case."""
    if subclss_tknzr.is_uncased:
        assert subclss_tknzr.norm(case_txt['input']) == case_txt['output']
    else:
        assert subclss_tknzr.norm(case_txt['input']) == case_txt['input']
