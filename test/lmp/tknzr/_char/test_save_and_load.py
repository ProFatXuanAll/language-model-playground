r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.load`.
- :py:meth:`lmp.tknzr.CharTknzr.save`.
"""

import json
import os

import pytest

from lmp.tknzr import CharTknzr


def test_save(
        char_tknzr: CharTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Test save operation for configuration file."""
    # Test Case: File exist.
    char_tknzr.save(exp_name)

    assert os.path.exists(file_path)

    # Test Case: File format.
    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Rasie error if file is invalid JSON.
        assert json.load(input_file)


def test_load(
        char_tknzr: CharTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Test load operation for configuration file."""
    # Test Case: Consistency file between save and load.
    char_tknzr.save(exp_name)

    load_tknzr = CharTknzr.load(exp_name)

    assert char_tknzr.__dict__ == load_tknzr.__dict__

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0, 1, 0.0, 0.1, 1.0, (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_exp_name in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            CharTknzr.load(exp_name=bad_exp_name)

        assert (
            '`exp_name` must be an instance of `str`' in str(excinfo.value)
        )
