r"""Test `lmp.util.load_config.`.

Usage:
    python -m unittest test/lmp/util/_config/test_load_config.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import gc
import inspect
import json
import math
import os
import unittest

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestLoadConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.load_config`."""

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

    def setUp(self):
        r"""Set up argparse namespace for config."""
        parser = argparse.ArgumentParser()

        # Required arguments.
        parser.add_argument(
            '--experiment',
            help='Current experiment name.',
            required=True,
            type=str
        )
        # Optional arguments.
        parser.add_argument(
            '--batch_size',
            default=32,
            help='Training batch size.',
            type=int
        )
        parser.add_argument(
            '--checkpoint',
            default=-1,
            help='Start from specific checkpoint.',
            type=int
        )
        parser.add_argument(
            '--checkpoint_step',
            default=500,
            help='Checkpoint save interval.',
            type=int
        )
        parser.add_argument(
            '--d_emb',
            default=100,
            help='Embedding dimension.',
            type=int
        )
        parser.add_argument(
            '--d_hid',
            default=300,
            help='Hidden dimension.',
            type=int
        )
        parser.add_argument(
            '--dataset',
            default='news_collection_title',
            help='Name of the dataset to perform experiment.',
            type=str
        )
        parser.add_argument(
            '--dropout',
            default=0.0,
            help='Dropout rate.',
            type=float
        )
        parser.add_argument(
            '--epoch',
            default=10,
            help='Number of training epochs.',
            type=int
        )
        parser.add_argument(
            '--is_uncased',
            action='store_true',
            help='Whether to convert text from upper cases to lower cases.'
        )
        parser.add_argument(
            '--learning_rate',
            default=1e-4,
            help='Gradient decent learning rate.',
            type=float
        )
        parser.add_argument(
            '--max_norm',
            default=1.0,
            help='Gradient bound to avoid gradient explosion.',
            type=float
        )
        parser.add_argument(
            '--max_seq_len',
            default=64,
            help='Text sample max length.',
            type=int
        )
        parser.add_argument(
            '--min_count',
            default=1,
            help='Filter out tokens occur less than `min_count`.',
            type=int
        )
        parser.add_argument(
            '--model_class',
            default='lstm',
            help="Language model's class.",
            type=str
        )
        parser.add_argument(
            '--num_linear_layers',
            default=2,
            help='Number of Linear layers.',
            type=int
        )
        parser.add_argument(
            '--num_rnn_layers',
            default=1,
            help='Number of rnn layers.',
            type=int
        )
        parser.add_argument(
            '--optimizer_class',
            default='adam',
            help="Optimizer's class.",
            type=str
        )
        parser.add_argument(
            '--seed',
            default=7,
            help='Control random seed.',
            type=int
        )
        parser.add_argument(
            '--tokenizer_class',
            default='whitespace_list',
            help="Tokenizer's class.",
            type=str
        )

        self.parser = parser

    def tearDown(self):
        r"""Delete `self.args`."""
        del self.parser
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_config),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='args',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=argparse.Namespace,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.config.BaseConfig
            ),
            msg=msg
        )

    def test_invalid_input_args(self):
        r"""Raise when `args` is invalid."""
        msg1 = 'Must raise `TypeError`  when `args` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_config(invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`args` must be an instance of `argparse.Namespace`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `lmp.config.BaseConfig`."""
        msg = 'Must return `lmp.config.BaseConfig`.'

        args = self.parser.parse_args(
            ['--experiment', 'util_config_test_case', ]
        )

        config = lmp.util.load_config(args)
        self.assertIsInstance(config, lmp.config.BaseConfig, msg=msg)

    def test_load_config_by_checkpoint(self):
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
                lmp.config.BaseConfig(**ans_attributes).save()
                self.assertTrue(os.path.exists(test_path), msg=msg1)

                args = self.parser.parse_args(
                    [
                        '--checkpoint', str(1),
                        '--epoch', str(ans_attributes['epoch']),
                        '--experiment', str(ans_attributes['experiment']),
                    ]
                )
                config = lmp.util.load_config(args)

                for attr_key, attr_value in ans_attributes.items():
                    self.assertTrue(hasattr(config, attr_key), msg=msg2)
                    self.assertIsInstance(
                        getattr(config, attr_key),
                        type(attr_value),
                        msg=msg2
                    )
                    self.assertEqual(
                        getattr(config, attr_key),
                        attr_value,
                        msg=msg2
                    )
            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
