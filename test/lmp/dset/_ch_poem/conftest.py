r"""Setup fixtures for testing :py:class:`lmp.dset.ChPoemDset`."""

import os

import pytest

import lmp.path
from lmp.dset import ChPoemDset


@pytest.fixture(autouse=True, scope='session')
def clean_dataset(request):
    r"""Clean the downloaded dataset at the end of test session."""

    def remove():
        for ver in ChPoemDset.vers:
            file_path = os.path.join(
                lmp.path.DATA_PATH,
                ChPoemDset.file_name.format(ver),
            )

            if os.path.exists(file_path):
                os.remove(file_path)

        # Remove data directory if it is empty.
        if not os.listdir(lmp.path.DATA_PATH):
            os.removedirs(lmp.path.DATA_PATH)

    request.addfinalizer(remove)
