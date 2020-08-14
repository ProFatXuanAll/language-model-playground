r"""Test `lmp.model.ResLSTMModel.predict`.

Usage:
    python -m unittest test/lmp/model/_res_lstm_model/test_predict.py
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

from lmp.model import ResLSTMModel


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.model.ResLSTMModel.predict`."""

    def setUp(self):
        r"""Set up hyper parameters and construct ResLSTMModel"""
        self.d_emb = 1
        self.d_hid = 1
        self.dropout = 0.1
        self.num_rnn_layers = 1
        self.num_linear_layers = 1
        self.pad_token_id = 0
        self.vocab_size = 10

        Parameters = (
            (
                ('d_emb', self.d_emb),
                ('d_hid', self.d_hid),
                ('dropout', self.dropout),
                ('num_rnn_layers', self.num_rnn_layers),
                ('num_linear_layers', self.num_linear_layers),
                ('pad_token_id', self.pad_token_id),
                ('vocab_size', self.vocab_size),
            ),
        )

        for parameters in Parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            self.models = [
                ResLSTMModel(*pos),
                ResLSTMModel(**kwargs),
            ]

    def tearDown(self):
        r"""Delete parameters and models."""
        del self.d_emb
        del self.d_hid
        del self.dropout
        del self.num_rnn_layers
        del self.num_linear_layers
        del self.pad_token_id
        del self.vocab_size
        del self.models
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(ResLSTMModel.predict),
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
        r"""Raise when `batch_sequences` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `batch_sequences` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            for model in self.models:
                with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                    model.predict(batch_sequences=invalid_input)

                if isinstance(ctx_man.exception, TypeError):
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_sequences` must be an instance of `Tensor`.',
                        msg=msg2
                    )

    def test_return_type(self):
        r"""Return `Tensor`."""
        msg = 'Must return `Tensor`.'
        examples = (
            torch.tensor(
                [
                    [1, 2],
                    [2, 3],
                    [3, 4]
                ]
            ),
        )

        for batch_sequences in examples:
            for model in self.models:
                pred_y = model.predict(batch_sequences)
                self.assertIsInstance(pred_y, torch.Tensor, msg=msg)


if __name__ == '__main__':
    unittest.main()
