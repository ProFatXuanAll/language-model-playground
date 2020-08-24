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
import os
import unittest

from itertools import product
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.config
import lmp.model
import lmp.path
import lmp.tokenizer
import lmp.util


class TestLoadModelByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.load_model_by_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.dataset = 'I-AM-TEST-DATASET'
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.checkpoint = 10
        cls.model_parameters = {
            'd_emb': [1, 2],
            'd_hid': [1, 2],
            'dropout': [0.0, 0.1],
            'is_uncased': [False, True],
            'model': [
                ('rnn', lmp.model.BaseRNNModel),
                ('gru', lmp.model.GRUModel),
                ('lstm', lmp.model.LSTMModel),
                ('res_rnn', lmp.model.BaseResRNNModel),
                ('res_gru', lmp.model.ResGRUModel),
                ('res_lstm', lmp.model.ResLSTMModel),
            ],
            'num_linear_layers': [1, 2],
            'num_rnn_layers': [1, 2],
            'tokenizer': [
                ('char_dict', lmp.tokenizer.CharDictTokenizer),
                ('char_list', lmp.tokenizer.CharListTokenizer),
                ('whitespace_dict', lmp.tokenizer.WhitespaceDictTokenizer),
                ('whitespace_list', lmp.tokenizer.WhitespaceListTokenizer),
            ],
        }
        cls.test_dir = os.path.join(lmp.path.DATA_PATH, cls.experiment)
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory and delete dynamic parameters."""
        os.removedirs(cls.test_dir)
        del cls.checkpoint
        del cls.dataset
        del cls.experiment
        del cls.model_parameters
        del cls.test_dir
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            dataset=self.__class__.dataset,
            experiment=self.__class__.experiment
        )
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete fixed parameters."""
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
        r"""Raise exception when input `checkpoint` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `checkpoint` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -2, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', (), [], {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
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
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be bigger than or equal to `-1`.',
                    msg=msg2
                )

    def test_invalid_input_config(self):
        r"""Raise `TypeError` when input `config` is invalid."""
        msg1 = 'Must raise `TypeError` when input `config` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`config` must be an instance of `lmp.config.BaseConfig`.',
                msg=msg2
            )

    def test_invalid_input_tokenizer(self):
        r"""Raise `TypeError` when input `tokenizer` is invalid."""
        msg1 = 'Must raise `TypeError` when input `tokenizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    tokenizer=invalid_input
                )

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

        test_path = os.path.join(
            self.__class__.test_dir,
            f'model-{self.__class__.checkpoint}.pt'
        )

        for (
                d_emb,
                d_hid,
                dropout,
                is_uncased,
                (model_class, model_cstr),
                num_linear_layers,
                num_rnn_layers,
                (tokenizer_class, tokenizer_cstr)
        ) in product(*self.__class__.model_parameters.values()):
            config = lmp.config.BaseConfig(
                d_emb=d_emb,
                d_hid=d_hid,
                dataset=self.__class__.dataset,
                dropout=dropout,
                experiment=self.__class__.experiment,
                model_class=model_class,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                tokenizer_class=tokenizer_class
            )
            tokenizer = tokenizer_cstr(is_uncased=is_uncased)
            model_1 = lmp.util.load_model_by_config(
                checkpoint=-1,
                config=config,
                tokenizer=tokenizer
            )

            self.assertIsInstance(model_1, model_cstr, msg=msg)

            try:
                # Create test file.
                torch.save(model_1.state_dict(), test_path)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                model_2 = lmp.util.load_model_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config,
                    tokenizer=tokenizer
                )

                self.assertIsInstance(model_2, model_cstr, msg=msg)
            finally:
                # Clean up test file.
                os.remove(test_path)

    def test_load_result(self):
        r"""Load result must be consistent."""
        msg = 'Inconsistent load result.'

        test_path = os.path.join(
            self.__class__.test_dir,
            f'model-{self.__class__.checkpoint}.pt'
        )

        for (
                d_emb,
                d_hid,
                dropout,
                is_uncased,
                (model_class, model_cstr),
                num_linear_layers,
                num_rnn_layers,
                (tokenizer_class, tokenizer_cstr)
        ) in product(*self.__class__.model_parameters.values()):
            config = lmp.config.BaseConfig(
                d_emb=d_emb,
                d_hid=d_hid,
                dataset=self.__class__.dataset,
                dropout=dropout,
                experiment=self.__class__.experiment,
                model_class=model_class,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                tokenizer_class=tokenizer_class
            )
            tokenizer = tokenizer_cstr(is_uncased=is_uncased)
            pad_token_id = tokenizer.convert_token_to_id(tokenizer.pad_token)
            vocab_size = tokenizer.vocab_size

            try:
                # Create test file.
                ans_model = model_cstr(
                    d_emb=d_emb,
                    d_hid=d_hid,
                    dropout=dropout,
                    num_linear_layers=num_linear_layers,
                    num_rnn_layers=num_rnn_layers,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size
                )
                torch.save(ans_model.state_dict(), test_path)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                model_1 = lmp.util.load_model_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config,
                    tokenizer=tokenizer
                )
                model_1 = model_1.to('cpu')

                model_2 = lmp.util.load_model_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config,
                    tokenizer=tokenizer
                )
                model_2 = model_2.to('cpu')

                self.assertEqual(
                    len(list(ans_model.parameters())),
                    len(list(model_1.parameters())),
                    msg=msg
                )
                self.assertEqual(
                    len(list(ans_model.parameters())),
                    len(list(model_2.parameters())),
                    msg=msg
                )

                for p1, p2 in zip(ans_model.parameters(),
                                  model_2.parameters()):
                    self.assertTrue((p1 == p2).all().item(), msg=msg)

            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
