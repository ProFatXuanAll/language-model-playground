r"""Test `lmp.model.BaseResRNNModel.predict`.

Usage:
    python -m unittest test.lmp.model._base_res_rnn_model.test_predict
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

# 3rd-party modules

import torch
import torch.nn

# self-made modules

from lmp.model import BaseResRNNModel


class TestPredict(unittest.TestCase):
    r"""Test case for `lmp.model.BaseResRNNModel.predict`."""

    @classmethod
    def setUpClass(cls):
        cls.batch_range = [1, 2]
        cls.d_emb_range = [1, 10]
        cls.d_hid_range = [1, 10]
        cls.dropout_range = [0.0, 0.1, 0.5, 1.0]
        cls.num_linear_layers_range = [1, 2]
        cls.num_rnn_layers_range = [1, 2]
        cls.pad_token_id_range = [0, 1, 2, 3]
        cls.sequence_range = list(range(1, 5))
        cls.vocab_size_range = [1, 5]

    @classmethod
    def tearDownClass(cls):
        del cls.batch_range
        del cls.d_emb_range
        del cls.d_hid_range
        del cls.dropout_range
        del cls.num_linear_layers_range
        del cls.num_rnn_layers_range
        del cls.pad_token_id_range
        del cls.sequence_range
        del cls.vocab_size_range
        gc.collect()

    def setUp(self):
        r"""Setup hyperparameters and construct `BaseResRNNModel`."""
        self.model_objs = []
        cls = self.__class__
        for d_emb in cls.d_emb_range:
            for d_hid in cls.d_hid_range:
                for dropout in cls.dropout_range:
                    for num_linear_layers in cls.num_linear_layers_range:
                        for num_rnn_layers in cls.num_rnn_layers_range:
                            for pad_token_id in cls.pad_token_id_range:
                                for vocab_size in cls.vocab_size_range:
                                    # skip invalid construct.
                                    if vocab_size <= pad_token_id:
                                        continue

                                    model = BaseResRNNModel(
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
            inspect.signature(BaseResRNNModel.predict),
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
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
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
