r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.load`.
- :py:meth:`lmp.tknzr.BaseTknzr.save`.
"""

import json
import os
from typing import Type

import pytest

from lmp.tknzr._base import BaseTknzr


def test_config_file_exist(
        exp_name: str,
        file_path: str,
        subclss_tknzr: BaseTknzr,
):
    r"""Save configuration as file."""
    subclss_tknzr.save(exp_name)
    assert os.path.exists(file_path)


def test_config_file_format(
        exp_name: str,
        file_path: str,
        subclss_tknzr: BaseTknzr,
):
    r"""Save configuration must be JSON format."""
    subclss_tknzr.save(exp_name)
    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if not valid JSON.
        assert json.load(input_file)


@pytest.mark.usefixtures('file_path')
def test_load_result(
        exp_name: str,
        subclss_tknzr: BaseTknzr,
        subclss_tknzr_clss: Type[BaseTknzr],
):
    r"""Ensure configuration consistency between save and load."""
    subclss_tknzr.save(exp_name)
    load_tknzr = subclss_tknzr_clss.load(exp_name)

    assert subclss_tknzr.is_uncased == load_tknzr.is_uncased
    assert subclss_tknzr.id2tk == load_tknzr.id2tk
    assert subclss_tknzr.max_vocab == load_tknzr.max_vocab
    assert subclss_tknzr.min_count == load_tknzr.min_count
    assert subclss_tknzr.tk2id == load_tknzr.tk2id
