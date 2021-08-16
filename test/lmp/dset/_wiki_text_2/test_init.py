r"""Test the construction of ChPoemDset

Test target:
- :py:meth:`lmp.tknzr._wiki_text_2.WikiText2Dset.init`.
"""

import os

import pytest 

from lmp.dset._wiki_text_2 import WikiText2Dset


def test_spls():
    r"""Test :py:attribute:`lmp.dset._wiki_text_2.WikiText2Dset.spls`"""

    wi_dset = WikiText2Dset()

    assert isinstance(wi_dset.spls, list)

def test_ver():
    r"""Test :py:attribute:`lmp.dset._wiki_text_2.WikiText2Dset.ver`"""

    wi_dset = WikiText2Dset()

    assert isinstance(wi_dset.ver, str)
