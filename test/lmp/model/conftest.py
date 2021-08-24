r"""Setup fixture for testing :py:mod:`lmp.model`."""

import os

import pytest

import lmp.path
from lmp.model._base import BaseModel


@pytest.fixture
def cleandir(request, ckpt: int, exp_name: str) -> str:
    r"""Clean model parameters output file and directories."""
    abs_dir_path = os.path.join(lmp.path.EXP_PATH, exp_name)
    abs_file_path = os.path.join(
        abs_dir_path, BaseModel.file_name.format(ckpt)
    )

    def remove():
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)

    if os.path.exists(abs_dir_path):
        os.removedirs(abs_dir_path)

    request.addfinalizer(remove)
