r"""Test text normalization.

Test target:
- :py:meth:`lmp.tknzr.util.norm`.
"""

from typing import Dict

import lmp.tknzr.util


def test_nfkc(non_nfkc_txt: Dict[str, str]):
    r"""Test output text is normalized with NFKC."""
    assert lmp.tknzr.util.norm(non_nfkc_txt['input']) == non_nfkc_txt['output']


def test_collapse_whitespace(cws_txt: Dict[str, str]):
    r"""Test output text collapse whitespaces."""
    assert lmp.tknzr.util.norm(cws_txt['input']) == cws_txt['output']


def test_strip_whitespace(htws_txt: Dict[str, str]):
    r"""Test output text is stripped."""
    assert lmp.tknzr.util.norm(htws_txt['input']) == htws_txt['output']
