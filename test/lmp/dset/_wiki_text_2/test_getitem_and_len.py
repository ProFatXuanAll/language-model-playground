r"""Test the abstract class methods

Test target:
- :py:meth:`lmp.dset._wiki_text_2.ChPoemDset.len`.
- :py:meth:`lmp.dset._wiki_text_2.ChPoemDset.getitem`.
"""
from lmp.dset._wiki_text_2 import WikiText2Dset
from test.lmp.dset._wiki_text_2.conftest import cleandir


def test_len_(dset_ver):
    r"""Check dataset size."""

    assert len(WikiText2Dset()) > 0
    cleandir(dset_ver)


def test_getitem_(dset_ver):
    r"""Check the sample of dataset."""

    assert isinstance(WikiText2Dset()[0], str)
    cleandir(dset_ver)
