r"""Test `lmp.util.load_model_by_config.`.

Usage:
    python -m unittest test.lmp.util._model.test_load_model_by_config
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import math
import unittest

from itertools import product
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestLoadModelByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.load_model_by_config`."""

    @classmethod
    def setUpClass(cls):
        cls.parameters = {
            'batch_size': [2, 4],
            'checkpoint_step': [100, 500],
            'd_emb': [5, 6],
            'd_hid': [7, 9],
            'dropout': [0.1, 0.5],
            'epoch': [10, 20],
            'is_uncased': [True, False],
            'learning_rate': [1e-4, 1e-5],
            'max_norm': [2.5, 6.0],
            'max_seq_len': [30, 45],
            'min_count': [2, 5],
            'model_class': [
                'rnn',
                'lstm',
                'gru',
                'res_rnn',
                'res_lstm',
                'res_gru'
            ],
            'num_linear_layers': [3, 6],
            'num_rnn_layers': [2, 5],
            'optimizer_class': ['sgd', 'adam'],
            'seed': [2, 4],
            'tokenizer_class': [
                'char_dict',
                'char_list',
                'whitespace_dict',
                'whitespace_list'
            ]
        }
        cls.param_values = [v for v in cls.parameters.values()]

    @classmethod
    def tearDownClass(cls):
        del cls.parameters
        del cls.param_values
        gc.collect()

    def setUp(self):
        r"""Set up parameters for `load_model_by_config`."""
        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            dataset='news_collection',
            experiment='util_load_model_by_config_unittest'
        )
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

        cls = self.__class__
        self.config_obj = []
        for (
            batch_size,
            checkpoint_step,
            d_emb,
            d_hid,
            dropout,
            epoch,
            is_uncased,
            learning_rate,
            max_norm,
            max_seq_len,
            min_count,
            model_class,
            num_linear_layers,
            num_rnn_layers,
            optimizer_class,
            seed,
            tokenizer_class
        ) in product(*cls.param_values):
            config = lmp.config.BaseConfig(
                batch_size=batch_size,
                checkpoint_step=checkpoint_step,
                d_emb=d_emb,
                d_hid=d_hid,
                dataset='news_collection',
                dropout=dropout,
                epoch=epoch,
                experiment='util_tokenizer_load_tokenizer_unittest',
                is_uncased=is_uncased,
                learning_rate=learning_rate,
                max_norm=max_norm,
                max_seq_len=max_seq_len,
                min_count=min_count,
                model_class=model_class,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                optimizer_class=optimizer_class,
                seed=seed,
                tokenizer_class=tokenizer_class
            )
            self.config_obj.append(config)

        self.tokenizer_obj = [
            lmp.tokenizer.CharDictTokenizer(True),
            lmp.tokenizer.CharDictTokenizer(),
            lmp.tokenizer.CharListTokenizer(True),
            lmp.tokenizer.CharListTokenizer(),
            lmp.tokenizer.WhitespaceDictTokenizer(True),
            lmp.tokenizer.WhitespaceDictTokenizer(),
            lmp.tokenizer.WhitespaceListTokenizer(True),
            lmp.tokenizer.WhitespaceListTokenizer(),
        ]

    def tearDown(self):
        r"""Delete parameters for `load_model_by_config`."""
        del self.checkpoint
        del self.config
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_model_by_config),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='config',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.config.BaseConfig,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=Union[
                    lmp.model.BaseRNNModel,
                    lmp.model.BaseResRNNModel
                ]
            ),
            msg=msg
        )

    def test_invalid_input_checkpoint(self):
        r"""Raise when `checkpoint` is invalid."""
        msg1 = 'Must raise `TypeError` when `checkpoint` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model_by_config(
                    checkpoint=invalid_input,
                    config=self.config,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
                )

    def test_invalid_input_config(self):
        r"""Raise when `config` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `config` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`config` must be an instance of `lmp.config.BaseConfig`.',
                    msg=msg2
                )

    def test_invalid_input_tokenizer(self):
        r"""Raise when `tokenizer` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `tokenizer` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    tokenizer=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokenizer` must be an instance of '
                    '`lmp.tokenizer.BaseTokenizer`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `lmp.model.BaseRNNModel` or `lmp.model.BaseResRNNModel`."""
        msg = (
            'Must return `lmp.model.BaseRNNModel` or '
            '`lmp.model.BaseResRNNModel`.'
        )

        examples = (
            (
                lmp.config.BaseConfig(
                    dataset='news_collection',
                    experiment='util_load_model_by_config_unittest',
                    model_class='gru',
                    tokenizer_class='char_dict'
                ),
                lmp.tokenizer.CharDictTokenizer(),
            ),
            (
                lmp.config.BaseConfig(
                    dataset='news_collection',
                    experiment='util_load_model_by_config_unittest',
                    model_class='res_lstm',
                    tokenizer_class='char_list'
                ),
                lmp.tokenizer.CharListTokenizer(is_uncased=True),
            ),
        )

        # examples = (
        #     (
        #         config,
        #         tokenizer,
        #     )
        #     for config in  self.config_obj
        #     for tokenizer in self.tokenizer_obj
        # )

        for config, tokenizer in examples:
            model = lmp.util.load_model_by_config(
                checkpoint=-1,
                config=config,
                tokenizer=tokenizer
            )

            try:
                self.assertIsInstance(model, lmp.model.BaseRNNModel, msg=msg)
            except AssertionError:
                self.assertIsInstance(
                    model, lmp.model.BaseResRNNModel, msg=msg)


if __name__ == '__main__':
    unittest.main()
