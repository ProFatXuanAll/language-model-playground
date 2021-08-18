r"""Test the downloaded file

Test target:
- :py:meth:`lmp.tknzr._ch_poem.ChPoemDset.download`.
"""
import os

from lmp.dset._ch_poem import ChPoemDset
from lmp import path
from test.lmp.dset._ch_poem.conftest import cleandir


def test_dset_file_exist(dset_ver):
    r"""Dataset must be downloaded to right places"""

    ch_dset = ChPoemDset()
    assert os.path.exists(path.DATA_PATH)
    assert os.path.exists(ch_dset.file_path)

    cleandir(dset_ver)
