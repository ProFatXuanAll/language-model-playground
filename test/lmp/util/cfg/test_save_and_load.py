r"""Test save and load operation for training configuration.

Test target:
- :py:meth:`lmp.util.cfg.load`.
- :py:meth:`lmp.util.cfg.save`.
"""

import argparse
import json
import os

import lmp
import lmp.util.cfg


def test_save(
    exp_name: str,
    clean_cfg,
):
    r"""Test save operation for training configuration file."""
    # Test Case: File exist.
    args = argparse.Namespace(a=1, b=2, c=3)
    file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
    file_path = os.path.join(file_dir, lmp.util.cfg.CFG_NAME)

    lmp.util.cfg.save(args, exp_name)

    assert os.path.exists(file_dir)
    assert os.path.exists(file_path)

    # Test Case: File format check.
    with open(file_path, 'r', encoding='utf-8') as input_file:
        # Raise error if file is invalid JSON.
        obj = json.load(input_file)
        assert isinstance(obj, dict)


def test_load(
    exp_name: str,
    clean_cfg,
):
    r"""Test load operation for training configuration file."""
    args = argparse.Namespace(a=1, b=2, c=3)

    lmp.util.cfg.save(args, exp_name)

    # Configuration value check.
    load_args = lmp.util.cfg.load(exp_name)

    assert args == load_args
