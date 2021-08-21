r"""Test the abstract class methods

Test target:
- :py:meth:`lmp.dset._ch_poem.ChPoemDset.len`.
- :py:meth:`lmp.dset._ch_poem.ChPoemDset.getitem`.
"""
from lmp.dset._ch_poem import ChPoemDset


def test_len_(dset_ver, download_dset, lastcleandir):
    r"""Check dataset size."""

    assert len(download_dset) > 0


def test_getitem_(dset_ver, download_dset, lastcleandir):
    r"""Check the sample of dataset."""

    assert isinstance(download_dset[0], str)
