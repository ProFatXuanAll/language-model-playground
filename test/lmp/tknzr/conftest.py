r"""Setup fixtures for testing :py:mod:`lmp.tknzr`."""

import os
from typing import Dict

import pytest

import lmp.path
from lmp.tknzr import BaseTknzr


@pytest.fixture(params=[
    {'input': 'ABC', 'output': 'abc'},
    {'input': 'abc', 'output': 'abc'},
])
def cased_txt(request) -> Dict[str, str]:
    r"""Text with whitespaces at head and tail."""
    return request.param


@pytest.fixture
def file_path(request, exp_name: str) -> str:
    r"""Tokenizer configuration file path.

    After testing, clean up files and directories create during test.
    """
    abs_dir_path = os.path.join(lmp.path.EXP_PATH, exp_name)
    abs_file_path = os.path.join(abs_dir_path, BaseTknzr.file_name)

    def fin():
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)
        if os.path.exists(abs_dir_path):
            os.removedirs(abs_dir_path)

    request.addfinalizer(fin)
    return abs_file_path
