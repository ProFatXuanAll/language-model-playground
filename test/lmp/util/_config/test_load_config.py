r"""Test `lmp.util.load_config.`.

Usage:
    python -m unittest test.lmp.util._config.test_load_config
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import gc
import inspect
import math
import os
import unittest

# self-made modules

import lmp.config
import lmp.model
import lmp.path
import lmp.util


class TestLoadConfig(unittest.TestCase):
    r"""Test case for `lmp.util.load_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and `config.json`."""
        cls.dataset = 'I-AM-A-TEST-DATASET'
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.test_dir = os.path.join(lmp.path.DATA_PATH, cls.experiment)
        os.makedirs(cls.test_dir)
        cls.config = lmp.config.BaseConfig(
            batch_size=1,
            checkpoint_step=1,
            d_emb=1,
            d_hid=1,
            dataset=cls.dataset,
            dropout=0.1,
            epoch=20,
            experiment=cls.experiment,
            learning_rate=1e-4,
            max_norm=1.0,
            max_seq_len=60,
            min_count=1,
            model_class='lstm',
            num_linear_layers=1,
            optimizer_class='adam',
            seed=1,
            tokenizer_class='char_dict',
        )
        cls.config.save()

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory and `config.json`."""
        os.remove(os.path.join(cls.test_dir, 'config.json'))
        os.removedirs(cls.test_dir)
        del cls.config
        del cls.dataset
        del cls.experiment
        del cls.test_dir
        gc.collect()

    def setUp(self):
        r"""Setup argparse namespace for config."""
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--experiment', type=str)
        self.parser.add_argument('--batch_size', type=int)
        self.parser.add_argument('--checkpoint', type=int)
        self.parser.add_argument('--checkpoint_step', type=int)
        self.parser.add_argument('--d_emb', type=int)
        self.parser.add_argument('--d_hid', type=int)
        self.parser.add_argument('--dataset', type=str)
        self.parser.add_argument('--dropout', type=float)
        self.parser.add_argument('--epoch', type=int)
        self.parser.add_argument('--is_uncased', action='store_true')
        self.parser.add_argument('--learning_rate', type=float)
        self.parser.add_argument('--max_norm', type=float)
        self.parser.add_argument('--max_seq_len', type=int)
        self.parser.add_argument('--min_count', type=int)
        self.parser.add_argument('--model_class', type=str)
        self.parser.add_argument('--num_linear_layers', type=int)
        self.parser.add_argument('--num_rnn_layers', type=int)
        self.parser.add_argument('--optimizer_class', type=str)
        self.parser.add_argument('--seed', type=int)
        self.parser.add_argument('--tokenizer_class', type=str)

    def tearDown(self):
        r"""Delete `self.parser`."""
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
        r"""Raise `TypeError` when input `args` is invalid."""
        msg1 = 'Must raise `TypeError` when input `args` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_config(invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`args` must be an instance of `argparse.Namespace`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `lmp.config.BaseConfig`."""
        msg = 'Must return `lmp.config.BaseConfig`.'
        examples = (
            [
                '--batch_size', str(1),
                '--checkpoint', str(1),
                '--checkpoint_step', str(500),
                '--d_emb', str(1),
                '--d_hid', str(1),
                '--dataset', self.__class__.dataset,
                '--dropout', str(0.1),
                '--epoch', str(20),
                '--experiment', self.__class__.experiment,
                '--learning_rate', str(1e-4),
                '--max_norm', str(1.0),
                '--max_seq_len', str(60),
                '--min_count', str(1),
                '--model_class', 'lstm',
                '--num_linear_layers', str(1),
                '--num_rnn_layers', str(1),
                '--optimizer_class', 'adam',
                '--seed', str(1),
                '--tokenizer_class', 'char_dict',
            ],
            [
                '--batch_size', str(101010),
                '--checkpoint', str(-1),
                '--checkpoint_step', str(999),
                '--d_emb', str(888),
                '--d_hid', str(777),
                '--dataset', 'world',
                '--dropout', str(0.69420),
                '--epoch', str(666),
                '--experiment', 'test',
                '--is_uncased',
                '--learning_rate', str(0.42069),
                '--max_norm', str(4.20),
                '--max_seq_len', str(555),
                '--min_count', str(444),
                '--model_class', 'hello world',
                '--num_linear_layers', str(333),
                '--num_rnn_layers', str(222),
                '--optimizer_class', 'WORLD',
                '--seed', str(111),
                '--tokenizer_class', 'HELLO',
            ],
        )

        for cmd_args in examples:
            self.assertIsInstance(
                lmp.util.load_config(self.parser.parse_args(cmd_args)),
                lmp.config.BaseConfig,
                msg=msg
            )

    def test_load_result(self):
        r"""Load result must be consistent."""
        msg = 'Inconsistent load result.'
        cls = self.__class__
        examples = (
            (
                [
                    '--batch_size', str(cls.config.batch_size),
                    '--checkpoint', str(1),
                    '--checkpoint_step', str(cls.config.checkpoint_step),
                    '--d_emb', str(cls.config.d_emb),
                    '--d_hid', str(cls.config.d_hid),
                    '--dataset', cls.config.dataset,
                    '--dropout', str(cls.config.dropout),
                    '--epoch', str(cls.config.epoch),
                    '--experiment', cls.config.experiment,
                    '--learning_rate', str(cls.config.learning_rate),
                    '--max_norm', str(cls.config.max_norm),
                    '--max_seq_len', str(cls.config.max_seq_len),
                    '--min_count', str(cls.config.min_count),
                    '--model_class', cls.config.model_class,
                    '--num_linear_layers', str(cls.config.num_linear_layers),
                    '--num_rnn_layers', str(cls.config.num_rnn_layers),
                    '--optimizer_class', cls.config.optimizer_class,
                    '--seed', str(cls.config.seed),
                    '--tokenizer_class', cls.config.tokenizer_class,
                ],
                {
                    'batch_size': cls.config.batch_size,
                    'checkpoint_step': 1,
                    'd_emb': cls.config.d_emb,
                    'd_hid': cls.config.d_hid,
                    'dataset': cls.config.dataset,
                    'dropout': cls.config.dropout,
                    'epoch': cls.config.epoch,
                    'experiment': cls.config.experiment,
                    'is_uncased': cls.config.is_uncased,
                    'learning_rate': cls.config.learning_rate,
                    'max_norm': cls.config.max_norm,
                    'max_seq_len': cls.config.max_seq_len,
                    'min_count': cls.config.min_count,
                    'model_class': cls.config.model_class,
                    'num_linear_layers': cls.config.num_linear_layers,
                    'num_rnn_layers': cls.config.num_rnn_layers,
                    'optimizer_class': cls.config.optimizer_class,
                    'seed': cls.config.seed,
                    'tokenizer_class': cls.config.tokenizer_class,
                },
            ),
            (
                [
                    '--batch_size', str(101010),
                    '--checkpoint', str(-1),
                    '--checkpoint_step', str(999),
                    '--d_emb', str(888),
                    '--d_hid', str(777),
                    '--dataset', 'world',
                    '--dropout', str(0.69420),
                    '--epoch', str(666),
                    '--experiment', 'test',
                    '--is_uncased',
                    '--learning_rate', str(0.42069),
                    '--max_norm', str(4.20),
                    '--max_seq_len', str(555),
                    '--min_count', str(444),
                    '--model_class', 'hello world',
                    '--num_linear_layers', str(333),
                    '--num_rnn_layers', str(222),
                    '--optimizer_class', 'WORLD',
                    '--seed', str(111),
                    '--tokenizer_class', 'HELLO',
                ],
                {
                    'batch_size': 101010,
                    'checkpoint_step': 999,
                    'd_emb': 888,
                    'd_hid': 777,
                    'dataset': 'world',
                    'dropout': 0.69420,
                    'epoch': 666,
                    'experiment': 'test',
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
            ),
        )

        for cmd_args, attributes in examples:
            config = lmp.util.load_config(self.parser.parse_args(cmd_args))

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


if __name__ == '__main__':
    unittest.main()
