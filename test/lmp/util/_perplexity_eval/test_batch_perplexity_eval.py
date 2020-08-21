r"""Test `lmp.util.batch_perplexity_eval.`.

Usage:
    python -m unittest test.lmp.util._perplexity_eval.test_batch_perplexity_eval
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
from typing import List
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestLoadOptimizerByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.batch_perplexity_eval`."""

    @classmethod
    def setUpClass(cls):
        cls.parameters = {
            'd_emb': [5, 6],
            'd_hid': [7, 9],
            'dropout': [0.1, 0.5],
            'num_linear_layers': [3, 6],
            'num_rnn_layers': [2, 5],
            'pad_token_id': [0, 1, 2, 3],
            'vocab_size': [10, 15]
        }
        cls.param_values = [v for v in cls.parameters.values()]

    @classmethod
    def tearDownClass(cls):
        del cls.parameters
        del cls.param_values
        gc.collect()

    def setUp(self):
        r"""Set up parameters for `batch_perplexity_eval`."""
        self.dataset = ['Today is Monday.', 'Tomorrow is Tuesday.']
        self.device = torch.tensor([10]).device
        self.model = lmp.model.BaseRNNModel(
            d_emb=4,
            d_hid=4,
            dropout=0.2,
            num_rnn_layers=1,
            num_linear_layers=1,
            pad_token_id=0,
            vocab_size=10
        )
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

        cls = self.__class__
        self.model_obj = []
        for (
            d_emb,
            d_hid,
            dropout,
            num_linear_layers,
            num_rnn_layers,
            pad_token_id,
            vocab_size
        ) in product(*cls.param_values):
            if vocab_size <= pad_token_id:
                continue
            model = lmp.model.BaseRNNModel(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )
            self.model_obj.append(model)

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
        r"""Delete parameters for `batch_perplexity_eval`."""
        del self.dataset
        del self.device
        del self.model
        del self.model_obj
        del self.tokenizer
        del self.tokenizer_obj
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.batch_perplexity_eval),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=List[str],
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
                return_annotation=List[float]
            ),
            msg=msg
        )

    def test_invalid_input_dataset(self):
        r"""Raise when `dataset` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `dataset` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.batch_perplexity_eval(
                    dataset=invalid_input,
                    device=self.device,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must be an instance of `List[str]`.',
                    msg=msg2
                )

    def test_invalid_input_device(self):
        r"""Raise when `device` is invalid."""
        msg1 = 'Must raise `TypeError` when `device` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.batch_perplexity_eval(
                    dataset=self.dataset,
                    device=invalid_input,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`device` must be an instance of `torch.device`.',
                    msg=msg2
                )
    
    def test_invalid_input_model(self):
        r"""Raise when `model` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `model` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.batch_perplexity_eval(
                    dataset=self.dataset,
                    device=self.device,
                    model=invalid_input,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model` must be an instance of '
                    '`Union['
                        'lmp.model.BaseRNNModel,'
                        'lmp.model.BaseResRNNModel'
                    ']`.',
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
                lmp.util.batch_perplexity_eval(
                    dataset=self.dataset,
                    device=self.device,
                    model=self.model,
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
        r"""Return `List[float]`."""
        msg = (
            'Must return `List[str]`.'
        )
        examples = (
            (
                tokenizer,
                model,
            )
            for tokenizer in self.tokenizer_obj
            for model in  self.model_obj
        )

        for tokenizer, model in examples:
            batch_ppls = lmp.util.batch_perplexity_eval(
                dataset=self.dataset,
                device=self.device,
                model=model,
                tokenizer=tokenizer
            )
            
            self.assertIsInstance(batch_ppls, List, msg=msg)
            for ppl in batch_ppls:
                self.assertIsInstance(ppl, float, msg=msg)


if __name__ == '__main__':
    unittest.main()
