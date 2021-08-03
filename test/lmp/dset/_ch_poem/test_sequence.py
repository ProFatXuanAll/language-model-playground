from lmp.dset._ch_poem import ChPoemDset


def test___len__():
    r"""Check dataset size.

    Dataset's size must be larger than 0
    """

    assert len(ChPoemDset()) > 0


def test___getitem__():
    r"""Check the sample of dataset."""

    assert isinstance(ChPoemDset()[0], str)
