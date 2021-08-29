r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.load`.
- :py:meth:`lmp.tknzr.CharTknzr.save`.
"""

import json
import os

from lmp.tknzr import CharTknzr


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
    r"""Saved configuration must be JSON format."""

    char_tknzr.save(exp_name)

    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if file is invalid JSON.
        assert json.load(input_file)


def test_load_result(
        char_tknzr: CharTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Ensure configuration consistency between save and load."""

    char_tknzr.save(exp_name)
    load_tknzr = CharTknzr.load(exp_name)

    assert char_tknzr.__dict__ == load_tknzr.__dict__
