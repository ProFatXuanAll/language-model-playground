r"""Test the construction of ChPoemDset

Test target:
- :py:meth:`lmp.tknzr._ch_poem.ChPoemDset.init`.
"""

import os

import pytest 

from lmp.dset._ch_poem import ChPoemDset


def test_spls():
    r"""Test :py:attribute:`lmp.dset._ch_poem.ChPoemDset.spls`"""

    ch_dset = ChPoemDset()

    assert isinstance(ch_dset.spls, list)


def test_ver():
    r"""Test :py:attribute:`lmp.dset._ch_poem.ChPoemDset.ver`"""

    ch_dset = ChPoemDset()

    assert isinstance(ch_dset.ver, str)