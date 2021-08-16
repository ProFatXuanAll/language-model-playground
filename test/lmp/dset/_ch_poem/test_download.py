r"""Test the downloaded file

Test target:
- :py:meth:`lmp.tknzr._ch_poem.ChPoemDset.download`.
"""

import os

import pytest 

from lmp.dset._ch_poem import ChPoemDset
from lmp import path


def test_dset_file_exist():
    r"""Dataset must be downloaded to right places"""

    ch_dset = ChPoemDset()

    ch_dset.download()

    file_name = ch_dset.file_name.format(ch_dset.ver)
    file_path = os.path.join(path.DATA_PATH, file_name)

    assert os.path.exists(file_path) == True
