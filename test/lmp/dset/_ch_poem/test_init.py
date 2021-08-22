r"""Test the construction of ChPoemDset

Test target:
- :py:meth:`lmp.tknzr._ch_poem.ChPoemDset.init`.
"""


def test_spls(download_dset, lastcleandir):
    r"""Test :py:attribute:`lmp.dset._ch_poem.ChPoemDset.spls`"""

    assert isinstance(download_dset.spls, list)


def test_ver(download_dset, lastcleandir):
    r"""Test :py:attribute:`lmp.dset._ch_poem.ChPoemDset.ver`"""

    assert isinstance(download_dset.ver, str)
