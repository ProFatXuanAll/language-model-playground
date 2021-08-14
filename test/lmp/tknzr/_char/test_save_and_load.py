r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.load`.
- :py:meth:`lmp.tknzr.CharTknzr.save`.
"""

import json
import os

import pytest

from lmp.tknzr._char import CharTknzr


def test_config_file_exist(
        char_tknzr: CharTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Save configuration as file."""

    char_tknzr.save(exp_name)

    assert os.path.exists(file_path)


def test_config_file_format(
        char_tknzr: CharTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Save configuration must be JSON format."""

    char_tknzr.save(exp_name)

    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if not valid JSON.
        assert json.load(input_file)


@pytest.mark.usefixtures('file_path')
def test_load_result(
        char_tknzr: CharTknzr,
        exp_name: str,
):
    r"""Ensure configuration consistency between save and load."""

    char_tknzr.save(exp_name)

    load_tknzr = char_tknzr.load(exp_name)

    assert char_tknzr.is_uncased == load_tknzr.is_uncased
    assert char_tknzr.id2tk == load_tknzr.id2tk
    assert char_tknzr.max_vocab == load_tknzr.max_vocab
    assert char_tknzr.min_count == load_tknzr.min_count
    assert char_tknzr.tk2id == load_tknzr.tk2id
