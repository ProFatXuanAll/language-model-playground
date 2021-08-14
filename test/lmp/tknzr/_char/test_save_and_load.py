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
@pytest.mark.parametrize(
    "parameters",
    [
        # Test tk2id is None
        #
        # Expect after save and load, the class has same attribute.
        {
            'is_uncased': False,
            'max_vocab': -1,
            'min_count': 1,
            'tk2id': None,
        },
        # Test tk2id is not empty
        #
        # Expect after save and load, the class has same attribute.
        {
            'is_uncased': False,
            'max_vocab': -1,
            'min_count': 1,
            'tk2id':
                {
                    '[bos]': 0,
                    '[eos]': 1,
                    '[pad]': 2,
                    '[unk]': 3,
                    'cc': 4,
                    'd': 5,
                    'b': 6,
                    'a': 7,
                },
        },
        # Test chinese characters
        #
        # Expect after save and load, the class has same attribute.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id':
                    {
                        '[bos]': 0,
                        '[eos]': 1,
                        '[pad]': 2,
                        '[unk]': 3,
                        '哈': 4,
                        '囉': 5,
                        '世': 6,
                        '界': 7,
                    },
            }
        ),
    ]
)
def test_load_result(
        parameters: dict,
        exp_name: str,
):
    r"""Ensure configuration consistency between save and load.

    Input differenct parameters to construct WsTknzr, and test
    the WsTknzr attribute.
    """

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    tknzr.save(exp_name)

    load_tknzr = tknzr.load(exp_name)

    assert tknzr.__dict__ == load_tknzr.__dict__
