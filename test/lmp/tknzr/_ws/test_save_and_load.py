r"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.load`.
- :py:meth:`lmp.tknzr.WsTknzr.save`.
"""

import json
import os

import pytest

from lmp.tknzr._ws import WsTknzr


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
    r"""Save configuration must be JSON format."""

    ws_tknzr.save(exp_name)

    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if not valid JSON.
        assert json.load(input_file)


@pytest.mark.usefixtures('file_path')
def test_load_result(
        ws_tknzr: WsTknzr,
        exp_name: str,
):
    r"""Ensure configuration consistency between save and load."""

    ws_tknzr.save(exp_name)

    load_tknzr = ws_tknzr.load(exp_name)

    assert ws_tknzr.is_uncased == load_tknzr.is_uncased
    assert ws_tknzr.id2tk == load_tknzr.id2tk
    assert ws_tknzr.max_vocab == load_tknzr.max_vocab
    assert ws_tknzr.min_count == load_tknzr.min_count
    assert ws_tknzr.tk2id == load_tknzr.tk2id
