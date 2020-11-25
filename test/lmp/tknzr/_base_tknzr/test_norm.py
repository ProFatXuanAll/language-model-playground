r"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.norm`.
"""

from typing import Dict

from lmp.tknzr._base import BaseTknzr


def test_nfkc(subclass_tknzr: BaseTknzr, non_nfkc_txt: Dict[str, str]):
    r"""Test output text is normalized with NFKC."""
    assert subclass_tknzr.norm(non_nfkc_txt['input']) == non_nfkc_txt['output']


def test_collapse_whitespace(
        subclass_tknzr: BaseTknzr,
        cws_txt: Dict[str, str]
):
    r"""Test output text collapse whitespaces."""
    assert subclass_tknzr.norm(cws_txt['input']) == cws_txt['output']


def test_strip_whitespace(subclass_tknzr: BaseTknzr, htws_txt: Dict[str, str]):
    r"""Test output text is stripped."""
    assert subclass_tknzr.norm(htws_txt['input']) == htws_txt['output']


def test_lower_case(subclass_tknzr: BaseTknzr, case_txt: Dict[str, str]):
    r"""Test output text is convert to lower case."""
    if subclass_tknzr.is_uncased:
        assert subclass_tknzr.norm(case_txt['input']) == case_txt['output']
    else:
        assert subclass_tknzr.norm(case_txt['input']) == case_txt['input']
