r"""Test `lmp.util.batch_perplexity_eval.`.

Usage:
    python -m unittest \
        test.lmp.util._perplexity_eval.test_batch_perplexity_eval
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
from typing import Iterable
from typing import List
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.model
import lmp.util


class TestLoadOptimizerByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.batch_perplexity_eval`."""

    @classmethod
    def setUpClass(cls):
        r"""Setup dynamic parameters."""
        cls.model_parameters = {
            'd_emb': [1, 2],
            'd_hid': [1, 2],
            'dataset': [
                ['hello'],
                ['hello', 'world'],
                ['hello', 'world', 'hello world'],
            ],
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
        self.dataset = ['hello', 'world', 'hello world']
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
        r"""Delete fixed parameters."""
        del self.dataset
        del self.device
        del self.model
        del self.tokenizer
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
                        annotation=Iterable[str],
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
                            lmp.model.BaseResRNNModel,
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
        r"""Raise `TypeError` when input `dataset` is invalid."""
        msg1 = 'Must raise `TypeError` when input `dataset` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ..., [False], [True], [0], [1], [-1], [0.0], [1.0],
            [math.nan], [-math.nan], [math.inf], [-math.inf], [0j], [1j],
            [b''], [()], [[]], [{}], [set()], [object()], [lambda x: x],
            [type], [None], [NotImplemented], [...], ['', False], ['', True],
            ['', 0], ['', 1], ['', -1], ['', 0.0], ['', 1.0], ['', math.nan],
            ['', -math.nan], ['', math.inf], ['', -math.inf], ['', 0j],
            ['', 1j], ['', b''], ['', ()], ['', []], ['', {}], ['', set()],
            ['', object()], ['', lambda x: x], ['', type], ['', None],
            ['', NotImplemented], ['', ...],
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.batch_perplexity_eval(
                    dataset=invalid_input,
                    device=self.device,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must be an instance of `Iterable[str]`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must not contain empty sequences.',
                    msg=msg2
                )

    def test_invalid_input_device(self):
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
                lmp.util.batch_perplexity_eval(
                    dataset=self.dataset,
                    device=invalid_input,
                    model=self.model,
                    tokenizer=self.tokenizer
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
                lmp.util.batch_perplexity_eval(
                    dataset=self.dataset,
                    device=self.device,
                    model=invalid_input,
                    tokenizer=self.tokenizer
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
                lmp.util.batch_perplexity_eval(
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
        r"""Return `List[float]`."""
        msg = 'Must return `List[float]`.'

        for (
                d_emb,
                d_hid,
                dataset,
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

            ppls = lmp.util.batch_perplexity_eval(
                dataset=dataset,
                device=torch.device('cpu'),
                model=model,
                tokenizer=tokenizer
            )

            self.assertIsInstance(ppls, list, msg=msg)
            for ppl in ppls:
                self.assertIsInstance(ppl, float, msg=msg)

    def test_return_value(self):
        r"""Perplexity is greater than or equal to zero."""
        msg = 'Perplexity must be greater than or equal to zero.'

        for (
                d_emb,
                d_hid,
                dataset,
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

            ppls = lmp.util.batch_perplexity_eval(
                dataset=dataset,
                device=torch.device('cpu'),
                model=model,
                tokenizer=tokenizer
            )

            for ppl in ppls:
                self.assertGreaterEqual(ppl, 0, msg=msg)

    def test_pure_function(self):
        r"""Perplexity must be the same when given the same input."""
        msg = 'Perplexity must be the same when given the same input'

        for (
                d_emb,
                d_hid,
                dataset,
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

            self.assertEqual(
                lmp.util.batch_perplexity_eval(
                    dataset=dataset,
                    device=torch.device('cpu'),
                    model=model,
                    tokenizer=tokenizer
                ),
                lmp.util.batch_perplexity_eval(
                    dataset=dataset,
                    device=torch.device('cpu'),
                    model=model,
                    tokenizer=tokenizer
                ),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
