r"""Test `lmp.util.analogy_inference`.

Usage:
    python -m unittest test.lmp.util._analogy_eval.test_analogy_inference
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

import lmp.model
import lmp.tokenizer

from lmp.util._analogy_eval import analogy_inference


class TestAnalogyInference(unittest.TestCase):
    r"""Test case of `lmp.util.analogy_inference"""

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
                lmp.model.BaseSelfAttentionRNNModel,
                lmp.model.SelfAttentionGRUModel,
                lmp.model.SelfAttentionLSTMModel,
                lmp.model.BaseSelfAttentionResRNNModel,
                lmp.model.SelfAttentionResGRUModel,
                lmp.model.SelfAttentionResLSTMModel,
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
        self.tokenizer = lmp.tokenizer.WhitespaceDictTokenizer()
        self.word_a = 'Taiwan'
        self.word_b = 'Taipei'
        self.word_c = 'Japan'

    def tearDown(self):
        r"""Delete fixed parameters"""
        del self.device
        del self.model
        del self.tokenizer
        del self.word_a
        del self.word_b
        del self.word_c
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'
        self.assertEqual(
            inspect.signature(lmp.util.analogy_inference),
            inspect.Signature(
                parameters=[
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
                            lmp.model.BaseResRNNModel,
                        ],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='word_a',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='word_b',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='word_c',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=str
            ),
            msg=msg
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
                lmp.util.analogy_inference(
                    device=invalid_input,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    word_a=self.word_a,
                    word_b=self.word_b,
                    word_c=self.word_c
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`device` must be an instance of `torch.device`.',
                msg=msg2
            )

    def test_invalid_input_model(self):
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
                lmp.util.analogy_inference(
                    device=self.device,
                    model=invalid_input,
                    tokenizer=self.tokenizer,
                    word_a=self.word_a,
                    word_b=self.word_b,
                    word_c=self.word_c
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`model` must be an instance of '
                '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.',
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
                lmp.util.analogy_inference(
                    device=self.device,
                    model=self.model,
                    tokenizer=invalid_input,
                    word_a=self.word_a,
                    word_b=self.word_b,
                    word_c=self.word_c
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
                msg=msg2
            )

    def test_invalid_input_word_a(self):
        r"""Raise `TypeError` when input `word_a` is invalid."""
        msg1 = 'Must raise `TypeError` when input `word_a` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.analogy_inference(
                    device=self.device,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    word_a=invalid_input,
                    word_b=self.word_b,
                    word_c=self.word_c
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`word_a` must be an instance of `str`.',
                msg=msg2
            )

    def test_invalid_input_word_b(self):
        r"""Raise `TypeError` when input `word_b` is invalid."""
        msg1 = 'Must raise `TypeError` when input `word_b` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.analogy_inference(
                    device=self.device,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    word_a=self.word_a,
                    word_b=invalid_input,
                    word_c=self.word_c
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`word_b` must be an instance of `str`.',
                msg=msg2
            )

    def test_invalid_input_word_c(self):
        r"""Raise `TypeError` when input `word_c` is invalid."""
        msg1 = 'Must raise `TypeError` when input `word_c` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.analogy_inference(
                    device=self.device,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    word_a=self.word_a,
                    word_b=self.word_b,
                    word_c=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`word_c` must be an instance of `str`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'

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

            pred_word = analogy_inference(
                device=self.device,
                model=model,
                tokenizer=tokenizer,
                word_a=self.word_a,
                word_b=self.word_b,
                word_c=self.word_c
            )
            self.assertIsInstance(pred_word, str, msg=msg)


if __name__ == '__main__':
    unittest.main()
