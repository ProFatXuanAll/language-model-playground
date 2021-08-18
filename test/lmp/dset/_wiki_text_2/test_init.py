r"""Test the construction of ChPoemDset

Test target:
- :py:meth:`lmp.tknzr._wiki_text_2.WikiText2Dset.init`.
"""
from lmp.dset._wiki_text_2 import WikiText2Dset
from test.lmp.dset._wiki_text_2.conftest import cleandir


def test_spls(dset_ver):
    r"""Test :py:attribute:`lmp.dset._wiki_text_2.WikiText2Dset.spls`"""

    wi_dset = WikiText2Dset()

    assert isinstance(wi_dset.spls, list)
    cleandir(dset_ver)


def test_ver(dset_ver):
    r"""Test :py:attribute:`lmp.dset._wiki_text_2.WikiText2Dset.ver`"""

    wi_dset = WikiText2Dset()

    assert isinstance(wi_dset.ver, str)

    cleandir(dset_ver)
