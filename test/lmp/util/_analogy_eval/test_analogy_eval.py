r"""Test `lmp.util.analogy_eval`.

Usage:
    python -m unittest test.lmp.util._analogy_eval.test_analogy_eval
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
from typing import Dict

# 3rd-party modules

import torch

# self-made modules

import lmp.model
from lmp.util._analogy_eval import analogy_eval
from lmp.dataset._analogy_dataset import AnalogyDataset


class TestAnalogyEval(unittest.TestCase):
    r"""Test case of `lmp.util.analogy_eval"""

    @classmethod
    def setUpClass(cls):
        r"""Setup dynamic parameters."""
        cls.model_parameters = {
            'd_emb': [1, 2],
            'd_hid': [1, 2],
            'dropout': [0.0, 0.1],
            'is_uncased': [False, True],
            'model_cstr': [
                lmp.model.BaseRNNModel,
                lmp.model.GRUModel,
                lmp.model.LSTMModel,
                lmp.model.BaseResRNNModel,
                lmp.model.ResGRUModel,
                lmp.model.ResLSTMModel,
            ],
            'num_linear_layers': [1, 2],
            'num_rnn_layers': [1, 2],
            'tokenizer_cstr': [
                lmp.tokenizer.CharDictTokenizer,
                lmp.tokenizer.CharListTokenizer,
                lmp.tokenizer.WhitespaceDictTokenizer,
                lmp.tokenizer.WhitespaceListTokenizer,
            ],
        }

    @classmethod
    def tearDownClass(cls):
        r"""Delete dynamic parameters."""
        del cls.model_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.dataset = lmp.dataset.AnalogyDataset(
            [
                [
                    'Taiwan',
                    'Taipei',
                    'Japan',
                    'Tokyo',
                    'capital'
                ]
            ]
        )
        self.device = torch.device('cpu')
        self.model = lmp.model.BaseRNNModel(
            d_emb=1,
            d_hid=1,
            dropout=0.0,
            num_linear_layers=1,
            num_rnn_layers=1,
            pad_token_id=0,
            vocab_size=5
        )
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete fixed parameters"""
        del self.device
        del self.model
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Emsure signature consistency."""
        msg = 'Inconsistent method signature.'
        self.assertEqual(
            inspect.signature(lmp.util.analogy_eval),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.dataset.AnalogyDataset,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='device',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.device,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='model',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Union[
                            lmp.model.BaseRNNModel,
                            lmp.model.BaseResRNNModel
                        ],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=Dict[str, float]
            ),
            msg=msg
        )

    def test_invaild_input_dataset(self):
        r"""Raise `TypeError` when input `dataset` is invalid."""
        msg1 = 'Must raise `TypeError` when input `dataset` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.analogy_eval(
                    dataset=invalid_input,
                    device=self.device,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`dataset` must be an instance of '
                '`lmp.dataset.AnalogyDataset`',
                msg=msg2
            )

    def test_invaild_input_device(self):
        r"""Raise `TypeError` when input `device` is invalid."""
        msg1 = 'Must raise `TypeError` when input `device` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.analogy_eval(
                    dataset=self.dataset,
                    device=invalid_input,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`device` must be an instance of `torch.device`.',
                msg=msg2
            )

    def test_invaild_input_model(self):
        r"""Raise `TypeError` when input `model` is invalid."""
        msg1 = 'Must raise `TypeError` when input `model` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.analogy_eval(
                    dataset=self.dataset,
                    device=self.device,
                    model=invalid_input,
                    tokenizer=self.tokenizer,
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`model` must be an instance of '
                '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.',
                msg=msg2
            )

    def test_invaild_input_tokenizer(self):
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
                lmp.util.analogy_eval(
                    dataset=self.dataset,
                    device=self.device,
                    model=self.model,
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `Dict[str, float]`."""
        msg = 'Must return `Dict[str, float]`.'

        for (
                d_emb,
                d_hid,
                dropout,
                is_uncased,
                model_cstr,
                num_linear_layers,
                num_rnn_layers,
                tokenizer_cstr,
        ) in product(*self.__class__.model_parameters.values()):
            tokenizer = tokenizer_cstr(is_uncased=is_uncased)
            pad_token_id = tokenizer.convert_token_to_id(tokenizer.pad_token)
            vocab_size = tokenizer.vocab_size
            model = model_cstr(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )

            acc_per_cat = analogy_eval(
                dataset=self.dataset,
                device=self.device,
                model=model,
                tokenizer=tokenizer
            )

            self.assertIsInstance(acc_per_cat, Dict, msg=msg)
            for category, score in acc_per_cat.items():
                self.assertIsInstance(category, str, msg=msg)
                self.assertIsInstance(score, float, msg=msg)

    def test_return_value(self):
        r"""Score in every category is greater than or equal to zero."""
        msg = 'Score in every category must greater than or equal to zero.'

        for (
            d_emb,
            d_hid,
            dropout,
            is_uncased,
            model_cstr,
            num_linear_layers,
            num_rnn_layers,
            tokenizer_cstr,
        ) in product(*self.__class__.model_parameters.values()):
            tokenizer = tokenizer_cstr(is_uncased=is_uncased)
            pad_token_id = tokenizer.convert_token_to_id(tokenizer.pad_token)
            vocab_size = tokenizer.vocab_size
            model = model_cstr(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )

            acc_per_cat = analogy_eval(
                dataset=self.dataset,
                device=self.device,
                model=model,
                tokenizer=tokenizer
            )

            for _, score in acc_per_cat.items():
                self.assertGreaterEqual(score, 0.0, msg=msg)


if __name__ == '__main__':
    unittest.main()
