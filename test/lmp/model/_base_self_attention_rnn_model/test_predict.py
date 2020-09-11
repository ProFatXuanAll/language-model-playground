r"""Test `lmp.model.BaseSelfAttentionRNNModel.predict`.

Usage:
    python -m unittest test.lmp.model._base_self_attention_rnn_model.test_predict
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

# 3rd-party modules

import torch
import torch.nn

# self-made modules

from lmp.model import BaseSelfAttentionRNNModel


class TestPredict(unittest.TestCase):
    r"""Test case for `lmp.model.BaseSelfAttentionRNNModel.predict`."""

    @classmethod
    def setUpClass(cls):
        cls.batch_range = [1, 2]
        cls.sequence_range = list(range(1, 5))
        cls.model_parameters = {
            'd_emb': [1, 2],
            'd_hid': [1, 2],
            'dropout': [0.0, 0.1],
            'num_linear_layers': [1, 2],
            'num_rnn_layers': [1, 2],
            'pad_token_id': [1, 2],
            'vocab_size': [1, 2],
        }

    @classmethod
    def tearDownClass(cls):
        del cls.batch_range
        del cls.sequence_range
        del cls.model_parameters
        gc.collect()

    def setUp(self):
        r"""Setup hyperparameters and construct `BaseSelfAttentionRNNModel`."""
        self.model_objs = []
        cls = self.__class__

        for (
            d_emb,
            d_hid,
            dropout,
            num_linear_layers,
            num_rnn_layers,
            pad_token_id,
            vocab_size
        ) in product(*cls.model_parameters.values()):
            # skip invalid construct.
            if vocab_size <= pad_token_id:
                continue
            model = BaseSelfAttentionRNNModel(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )
            self.model_objs.append({
                'd_emb': d_emb,
                'd_hid': d_hid,
                'dropout': dropout,
                'model': model,
                'num_linear_layers': num_linear_layers,
                'num_rnn_layers': num_rnn_layers,
                'pad_token_id': pad_token_id,
                'vocab_size': vocab_size,
            })

    def tearDown(self):
        r"""Delete model instances."""
        del self.model_objs
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseSelfAttentionRNNModel.predict),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='batch_sequences',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.Tensor,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=torch.Tensor
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequences(self):
        r"""Raise `TypeError` when input `batch_sequences` is invalid."""
        msg = 'Must raise `TypeError` when input `batch_sequences` is invalid.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            for model_obj in self.model_objs:
                model = model_obj['model']
                with self.assertRaises(TypeError, msg=msg):
                    model.predict(batch_sequences=invalid_input)

    def test_return_type(self):
        r"""Return `Tensor`."""
        msg = 'Must return `Tensor`.'

        examples = (
            (
                torch.randint(
                    0,
                    model_obj['vocab_size'],
                    (batch_size, sequence_len)
                ),
                model_obj['model'],
            )
            for batch_size in self.__class__.batch_range
            for sequence_len in self.__class__.sequence_range
            for model_obj in self.model_objs
        )

        for x, model in examples:
            self.assertIsInstance(model.predict(x), torch.Tensor, msg=msg)

    def test_return_size(self):
        r"""Test return size"""
        msg = 'Return size must be {}.'

        examples = (
            (
                torch.randint(
                    0,
                    model_obj['vocab_size'],
                    (batch_size, sequence_len)
                ),
                model_obj['model'],
                model_obj['vocab_size']
            )
            for batch_size in self.__class__.batch_range
            for sequence_len in self.__class__.sequence_range
            for model_obj in self.model_objs
        )

        for x, model, vocab_size in examples:
            pred = model.predict(x)
            for s1, s2 in zip(x.size(), pred.size()):
                self.assertEqual(s1, s2, msg=msg)
            self.assertEqual(pred.size(-1), vocab_size)


if __name__ == '__main__':
    unittest.main()
