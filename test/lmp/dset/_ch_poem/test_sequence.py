from lmp.dset._ch_poem import ChPoemDset


def test_len_():
    r"""Check dataset size."""

    assert len(ChPoemDset()) > 0


def test_getitem_():
    r"""Check the sample of dataset."""

    assert isinstance(ChPoemDset()[0], str)
