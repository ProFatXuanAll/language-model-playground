r"""Test input sequence normalization.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.norm`.
"""

from typing import Dict

from lmp.tknzr._base_tknzr import BaseTknzr


def test_nfkc(subclass_tknzr: BaseTknzr, non_nfkc_seq: Dict[str, str]):
    r"""Test output sequence is normalized with NFKC."""
    assert subclass_tknzr.norm(non_nfkc_seq['input']) == non_nfkc_seq['output']


def test_collapse_whitespace(
        subclass_tknzr: BaseTknzr,
        cws_seq: Dict[str, str]
):
    r"""Test output sequence collapse whitespaces."""
    assert subclass_tknzr.norm(cws_seq['input']) == cws_seq['output']


def test_strip_whitespace(subclass_tknzr: BaseTknzr, htws_seq: Dict[str, str]):
    r"""Test output sequence is stripped."""
    assert subclass_tknzr.norm(htws_seq['input']) == htws_seq['output']


def test_lower_case(subclass_tknzr: BaseTknzr, case_seq: Dict[str, str]):
    r"""Test output sequence is convert to lower case."""
    if subclass_tknzr.is_uncased:
        assert subclass_tknzr.norm(case_seq['input']) == case_seq['output']
    else:
        assert subclass_tknzr.norm(case_seq['input']) == case_seq['input']
