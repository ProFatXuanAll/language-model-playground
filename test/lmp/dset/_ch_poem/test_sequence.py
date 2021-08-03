from lmp.dset._ch_poem import ChPoemDset


def test___len__():
    r"""Check dataset size.

    Dataset's size is larger than 0
    """

    dset = ChPoemDset()
    assert len(dset) > 0


def test___getitem__():
    r"""Check the sample of dataset.

    Dataset type must be correct.
    """
    dset = ChPoemDset()
    assert isinstance(dset[0], str)
