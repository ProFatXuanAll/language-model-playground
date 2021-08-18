r"""Setup fixture for testing :py:mod:`lmp.dset._ch_poem.ChPoemDset`."""

import os

import pytest

from lmp import path


@pytest.fixture
def dset_ver():
    """Version of dataset"""

    return '唐'


def cleandir(dset_ver):
    r"""Clean the downloaded dataset"""

    dset_name = '{}.csv.zip'.format(dset_ver)
    file_path = os.path.join(path.DATA_PATH, dset_name)

    if os.path.exists(file_path):
        os.remove(file_path)
