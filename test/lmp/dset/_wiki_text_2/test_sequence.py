from lmp.dset._wiki_text_2 import WikiText2Dset


def test_len_():
    r"""Check dataset size."""

    assert len(WikiText2Dset()) > 0


def test_getitem_():
    r"""Check the sample of dataset."""

    assert isinstance(WikiText2Dset()[0], str)
