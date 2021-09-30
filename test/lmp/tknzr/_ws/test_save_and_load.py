r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.load`.
- :py:meth:`lmp.tknzr.WsTknzr.save`.
"""

import json
import os

from lmp.tknzr import WsTknzr


def test_save(
        ws_tknzr: WsTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Test save operation for configuration file."""
    # Test Case: File exist.
    ws_tknzr.save(exp_name)

    assert os.path.exists(file_path)

    # Test Case: File format.
    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Rasie error if file is invalid JSON.
        assert json.load(input_file)


def test_load(
        ws_tknzr: WsTknzr,
        exp_name: str,
        file_path: str,
):
    r"""Test load operation for configuration file."""
    # Test Case: Consistency file between save and load.
    ws_tknzr.save(exp_name)

    load_tknzr = ws_tknzr.load(exp_name)

    assert ws_tknzr.__dict__ == load_tknzr.__dict__
