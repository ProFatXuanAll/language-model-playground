from lmp.dset._wiki_text_2 import WikiText2Dset


def test___len__():
    r"""Check dataset size.

    Dataset's size must be larger than 0
    """

    assert len(WikiText2Dset()) > 0


def test___getitem__():
    r"""Check the sample of dataset."""

    assert isinstance(WikiText2Dset()[0], str)
