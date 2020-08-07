r"""Test `lmp.config.BaseConfig.load`.

Usage:
    python -m unittest test/lmp/config/test_load.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import json
import math
import os
import unittest

# self-made modules

import lmp.path

from lmp.config import BaseConfig


class TestLoad(unittest.TestCase):
    r"""Test Case for `lmp.config.BaseConfig.load`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory."""
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.test_dir = os.path.join(
            lmp.path.DATA_PATH,
            cls.experiment
        )
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory."""
        os.removedirs(cls.test_dir)

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseConfig.load),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_experiment(self):
        r"""Raise when `experiment` is invalid."""
        msg1 = (
            'Must raise `FileNotFoundError`, `TypeError` or `ValueError` when '
            '`experiment` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', 'I-DO-NOT-EXIST', b'', [], (), {}, set(),
            object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (FileNotFoundError, TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseConfig.load(experiment=invalid_input)

            if isinstance(ctx_man.exception, FileNotFoundError):
                file_path = os.path.join(
                    lmp.path.DATA_PATH,
                    invalid_input,
                    'config.json'
                )
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'file {file_path} does not exist.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must be instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must not be empty.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_json(self):
        r"""Raise when configuration is invalid."""
        msg = (
            'Must raise `JSONDecodeError` when configuration is not in JSON '
            'format.'
        )

        test_path = os.path.join(self.__class__.test_dir, 'config.json')

        try:
            # Create test file.
            with open(test_path, 'w', encoding='utf-8') as output_file:
                output_file.write('Invalid JSON format.')

            with self.assertRaises(json.JSONDecodeError, msg=msg):
                BaseConfig.load(experiment=self.__class__.experiment)
        finally:
            # Clean up test file.
            os.remove(test_path)

    def test_load_result(self):
        r"""Load result must be consistent."""
        msg = 'Inconsistent load result.'
        examples = (
            {
                'batch_size': 111,
                'checkpoint_step': 222,
                'd_emb': 333,
                'd_hid': 444,
                'dataset': 'hello',
                'dropout': 0.42069,
                'epoch': 555,
                'experiment': 'world',
                'is_uncased': True,
                'learning_rate': 0.69420,
                'max_norm': 6.9,
                'max_seq_len': 666,
                'min_count': 777,
                'model_class': 'HELLO',
                'num_linear_layers': 888,
                'num_rnn_layers': 999,
                'optimizer_class': 'WORLD',
                'seed': 101010,
                'tokenizer_class': 'hello world',
            },
            {
                'batch_size': 101010,
                'checkpoint_step': 999,
                'd_emb': 888,
                'd_hid': 777,
                'dataset': 'world',
                'dropout': 0.69420,
                'epoch': 666,
                'experiment': 'hello',
                'is_uncased': True,
                'learning_rate': 0.42069,
                'max_norm': 4.20,
                'max_seq_len': 555,
                'min_count': 444,
                'model_class': 'hello world',
                'num_linear_layers': 333,
                'num_rnn_layers': 222,
                'optimizer_class': 'WORLD',
                'seed': 111,
                'tokenizer_class': 'HELLO',
            },
        )

        for attributes in examples:
            test_path = os.path.join(
                self.__class__.test_dir,
                'config.json'
            )

            try:
                # Create test file.
                with open(test_path, 'w', encoding='utf-8') as output_file:
                    json.dump(attributes, output_file)

                config = BaseConfig.load(experiment=self.__class__.experiment)
                for attr_key, attr_value in attributes.items():
                    self.assertTrue(hasattr(config, attr_key), msg=msg)
                    self.assertIsInstance(
                        getattr(config, attr_key),
                        type(attr_value),
                        msg=msg
                    )
                    self.assertEqual(
                        getattr(config, attr_key),
                        attr_value,
                        msg=msg
                    )
            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
