r"""Setup fixture for testing :py:mod:`lmp.tknzr`."""

import os
from typing import Dict, Union

import pytest

import lmp.path
from lmp.tknzr._base import BaseTknzr


@pytest.fixture(params=[True, False])
def is_uncased(request) -> bool:
    r"""Tokenizer instance attribute ``is_uncased``."""
    return request.param


@pytest.fixture(params=[1, 2])
def min_count(request) -> int:
    r"""Tokenizer instance attribute ``min_count``."""
    return request.param


@pytest.fixture(params=[1, 2])
def max_vocab(request) -> int:
    r"""Tokenizer instance attribute ``max_vocab``."""
    return request.param


@pytest.fixture(params=[
    None,
    {
        '[bos]': 0,
        '[eos]': 1,
        '[pad]': 2,
        '[unk]': 3,
        'a': 4,
        'b': 5,
        'c': 6,
    },
])
def tk2id(request) -> Union[None, Dict[str, int]]:
    r"""Tokenizer instance attribute ``tk2id``."""
    return request.param


@pytest.fixture(params=[
    {'input': 'ABC', 'output': 'abc'},
    {'input': 'abc', 'output': 'abc'},
])
def case_txt(request) -> Dict[str, str]:
    r"""Text with whitespaces at head and tail."""
    return request.param


@pytest.fixture
def file_path(request, exp_name: str) -> str:
    r"""Tokenizer configuration output file path.

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
