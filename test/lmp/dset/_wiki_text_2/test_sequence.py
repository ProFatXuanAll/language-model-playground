from lmp.dset._wiki_text_2 import WikiText2Dset


def test___len__():
    r"""Check dataset size.

    Dataset's size is larger than 0
    """

    dset = WikiText2Dset()
    assert len(dset) > 0


def test___getitem__():
    r"""Check the sample of dataset.

    Dataset type must be correct.
    """
    dset = WikiText2Dset()
    assert isinstance(dset[0], str)
