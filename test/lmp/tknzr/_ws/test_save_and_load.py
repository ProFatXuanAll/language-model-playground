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
        exp_name: str,
        file_path: str,
):
    r"""Save configuration as file."""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
    )

    tknzr.save(exp_name)

    assert os.path.exists(file_path)


def test_config_file_format(
        exp_name: str,
        file_path: str,
):
    r"""Save configuration must be JSON format."""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
    )

    tknzr.save(exp_name)

    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if not valid JSON.
        assert json.load(input_file)


@pytest.mark.usefixtures('file_path')
def test_load_result(
        exp_name: str,
):
    r"""Ensure configuration consistency between save and load."""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
    )

    tknzr.save(exp_name)

    load_tknzr = tknzr.load(exp_name)

    assert tknzr.is_uncased == load_tknzr.is_uncased
    assert tknzr.id2tk == load_tknzr.id2tk
    assert tknzr.max_vocab == load_tknzr.max_vocab
    assert tknzr.min_count == load_tknzr.min_count
    assert tknzr.tk2id == load_tknzr.tk2id
