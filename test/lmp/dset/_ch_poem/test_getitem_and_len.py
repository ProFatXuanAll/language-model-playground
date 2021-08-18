r"""Test the abstract class methods

Test target:
- :py:meth:`lmp.dset._ch_poem.ChPoemDset.len`.
- :py:meth:`lmp.dset._ch_poem.ChPoemDset.getitem`.
"""
from test.lmp.dset._ch_poem.conftest import cleandir
from lmp.dset._ch_poem import ChPoemDset


def test_len_(dset_ver):
    r"""Check dataset size."""

    assert len(ChPoemDset()) > 0
    cleandir(dset_ver)


def test_getitem_(dset_ver):
    r"""Check the sample of dataset."""

    assert isinstance(ChPoemDset()[0], str)
    cleandir(dset_ver)
