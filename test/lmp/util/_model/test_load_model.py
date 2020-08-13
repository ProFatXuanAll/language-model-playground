r"""Test `lmp.util.load_model.`.

Usage:
    python -m unittest test/lmp/util/_model/test_load_model.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import inspect
import json
import math
import os
import unittest

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestLoadModel(unittest.TestCase):
    r"""Test Case for `lmp.util.load_model`."""

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
        r"""Set up parameters for `load_model`."""
        self.checkpoint = -1
        self.d_emb = 1
        self.d_hid = 1
        self.device =
        self.dropout = 0.1
        self.experiment = 'test_util_load_model'
        self.model_class = 'res_rnn'
        self.num_linear_layers = 1
        self.num_rnn_layers = 1
        self.pad_token_id = 0
        self.vocab_size = 10

    def setUp(self):
        r"""Set up argparse namespace for config."""
        args = {
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
        }
        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(**args)
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete `self.args`."""
        del self.checkpoint

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_model),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='d_emb',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='d_hid',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='device',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.device,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='dropout',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='model_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='num_linear_layers',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.config.BaseConfig
            ),
            msg=msg
        )

        def test_invalid_input_d_emb(self):
        r"""Raise when `d_emb` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_emb` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseRNNModel(
                    d_emb=invalid_input,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_checkpoint(self):
        r"""Raise when `checkpoint` is invalid."""
        msg1 = 'Must raise `TypeError` when `checkpoint` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model(
                    checkpoint=invalid_input,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_d_emb(self):
        r"""Raise when `d_emb` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_emb` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseRNNModel(
                    d_emb=invalid_input,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_d_hid(self):
        r"""Raise when `d_hid` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_hid` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseRNNModel(
                    d_emb=1,
                    d_hid=invalid_input,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_dropout(self):
        r"""Raise when `dropout` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `dropout` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, -1.0, 1.1, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseRNNModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=invalid_input,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )
            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must be instance of `float`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must range from `0.0` to `1.0`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_experiment(self):
        r"""Raise when `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `experiment` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseConfig(dataset='test', experiment=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
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

    def test_invalid_input_model_class(self):

    def test_invalid_input_num_linear_layers(self):
        r"""Raise when `num_linear_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `num_linear_layers` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseRNNModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=invalid_input,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_linear_layers` must be instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_linear_layers` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_return_type(self):
        r"""Return `lmp.config.BaseConfig`."""
        msg = 'Must return `lmp.config.BaseConfig`.'

        args = self.parser.parse_args(
            ['--experiment', 'util_config_test_case', ]
        )

        config = lmp.util.load_model(args)
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
                config = lmp.util.load_model(args)

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
