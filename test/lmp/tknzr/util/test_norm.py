r"""Test input sequence normalization.

Test target:
- :py:meth:`lmp.tknzr.util.norm`.
"""

from typing import Dict

import lmp.tknzr.util


def test_nfkc(non_nfkc_seq: Dict[str, str]):
    r"""Test output sequence is normalized with NFKC."""
    assert lmp.tknzr.util.norm(non_nfkc_seq['input']) == non_nfkc_seq['output']


def test_collapse_whitespace(cws_seq: Dict[str, str]):
    r"""Test output sequence collapse whitespaces."""
    assert lmp.tknzr.util.norm(cws_seq['input']) == cws_seq['output']


def test_strip_whitespace(htws_seq: Dict[str, str]):
    r"""Test output sequence is stripped."""
    assert lmp.tknzr.util.norm(htws_seq['input']) == htws_seq['output']
