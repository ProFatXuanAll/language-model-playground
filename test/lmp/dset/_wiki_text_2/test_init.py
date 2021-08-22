r"""Test the construction of ChPoemDset

Test target:
- :py:meth:`lmp.tknzr._wiki_text_2.WikiText2Dset.init`.
"""


def test_spls(download_dset, lastcleandir):
    r"""Test :py:attribute:`lmp.dset._wiki_text_2.WikiText2Dset.spls`"""
    assert isinstance(download_dset.spls, list)


def test_ver(download_dset, lastcleandir):
    r"""Test :py:attribute:`lmp.dset._wiki_text_2.WikiText2Dset.ver`"""
    assert isinstance(download_dset.ver, str)
