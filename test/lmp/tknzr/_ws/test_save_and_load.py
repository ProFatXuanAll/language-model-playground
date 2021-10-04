r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.load`.
- :py:meth:`lmp.tknzr.WsTknzr.save`.
"""

import json
import os

import pytest

from lmp.tknzr import WsTknzr


def test_config_file_exist(
        ws_tknzr: WsTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Save configuration as file."""

    ws_tknzr.save(exp_name)

    assert os.path.exists(file_path)


def test_config_file_format(
        ws_tknzr: WsTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Saved configuration must be JSON format."""

    ws_tknzr.save(exp_name)

    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if file is invalid JSON.
        assert json.load(input_file)


def test_load_result(
        ws_tknzr: WsTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Ensure configuration consistency between save and load."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        False, True, 0, 1, 0.0, 0.1, 1.0, (), [], {}, set(), None, ...,
        NotImplemented,
    ]

    for bad_exp_name in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            WsTknzr.load(exp_name=bad_exp_name)

        assert (
            '`exp_name` must be an instance of `str`' in str(excinfo.value)
        )

    # Test case: Valid input.
    ws_tknzr.save(exp_name)
    load_tknzr = WsTknzr.load(exp_name)

    assert ws_tknzr.__dict__ == load_tknzr.__dict__
