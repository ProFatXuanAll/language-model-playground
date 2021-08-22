r"""Test the abstract class methods

Test target:
- :py:meth:`lmp.dset._wiki_text_2.ChPoemDset.len`.
- :py:meth:`lmp.dset._wiki_text_2.ChPoemDset.getitem`.
"""


def test_len_(download_dset, lastcleandir):
    r"""Check dataset size."""

    assert len(download_dset) > 0


def test_getitem_(download_dset, lastcleandir):
    r"""Check the sample of dataset."""

    assert isinstance(download_dset[0], str)
