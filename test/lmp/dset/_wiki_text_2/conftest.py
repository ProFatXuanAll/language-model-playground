r"""Setup fixture for testing :py:mod:`lmp.dset._ch_poem.ChPoemDset`."""
import os

import pytest

from lmp import path
from lmp.dset._wiki_text_2 import WikiText2Dset


@pytest.fixture(
    params =      
    [
        'test', 'train', 'valid'
    ],
)
def dset_ver(request):
    """Version of dataset"""

    return request.param



@pytest.fixture
def download_dset(dset_ver):
    r"""Download and return ChPoemDset in the function scope"""
    wi_dset = WikiText2Dset(ver = dset_ver)
    
    return wi_dset


@pytest.fixture
def cleandir(dset_ver, download_dset, request):
    r"""Clean the downloaded dataset in the middle of testing"""

    def remove():
        file_path = os.path.join(
                        path.DATA_PATH, 
                        download_dset.file_name.format(dset_ver),
                    )

        if os.path.exists(file_path):
            os.remove(file_path)

    request.addfinalizer(remove)


@pytest.fixture(scope = "session")
def lastcleandir(request):
    r"""Clean the downloaded dataset at the end of testing session"""

    def remove():
        dset_list = [ 
            'test', 'train', 'valid'
        ]

        for i in dset_list:
            file_path = os.path.join(
                            path.DATA_PATH, 
                            WikiText2Dset(ver = i).file_name.format(i),
                        )

            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(path.DATA_PATH):
                os.removedirs(path.DATA_PATH)

    request.addfinalizer(remove)
