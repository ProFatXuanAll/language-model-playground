r"""Test `lmp.config.BaseConfig.save`.

Usage:
    python -m unittest test/lmp/config/test_save.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import json
import os
import unittest

# self-made modules

import lmp.path

from lmp.config import BaseConfig


class TestSave(unittest.TestCase):
    r"""Test case for `lmp.config.BaseConfig.save`."""

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
        del cls.test_dir
        del cls.experiment
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseConfig.save),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_save_result(self):
        r"""Save result must be consistent."""
        msg1 = 'Must save as `config.json`.'
        msg2 = 'Inconsistent save result.'
        examples = (
            {
                'batch_size': 111,
                'checkpoint_step': 222,
                'd_emb': 333,
                'd_hid': 444,
                'dataset': 'hello',
                'dropout': 0.42069,
                'epoch': 555,
                'experiment': self.__class__.experiment,
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
                'experiment': self.__class__.experiment,
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

        for ans_attributes in examples:
            test_path = os.path.join(
                self.__class__.test_dir,
                'config.json'
            )

            try:
                # Create test file.
                BaseConfig(**ans_attributes).save()
                self.assertTrue(os.path.exists(test_path), msg=msg1)

                with open(test_path, 'r') as input_file:
                    attributes = json.load(input_file)

                for attr_key, attr_value in attributes.items():
                    self.assertIn(attr_key, ans_attributes, msg=msg2)
                    self.assertIsInstance(
                        ans_attributes[attr_key],
                        type(attr_value),
                        msg=msg2
                    )
                    self.assertEqual(
                        ans_attributes[attr_key],
                        attr_value,
                        msg=msg2
                    )
            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
