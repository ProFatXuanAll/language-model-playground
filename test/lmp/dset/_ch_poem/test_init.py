r"""Test the construction of ChPoemDset

Test target:
- :py:meth:`lmp.tknzr._ch_poem.ChPoemDset.init`.
"""
from test.lmp.dset._ch_poem.conftest import cleandir
from lmp.dset._ch_poem import ChPoemDset


def test_spls(dset_ver):
    r"""Test :py:attribute:`lmp.dset._ch_poem.ChPoemDset.spls`"""

    ch_dset = ChPoemDset()

    assert isinstance(ch_dset.spls, list)
    cleandir(dset_ver)


def test_ver(dset_ver):
    r"""Test :py:attribute:`lmp.dset._ch_poem.ChPoemDset.ver`"""

    ch_dset = ChPoemDset()

    assert isinstance(ch_dset.ver, str)
    cleandir(dset_ver)
