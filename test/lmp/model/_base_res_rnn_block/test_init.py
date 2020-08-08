r"""Test `lmp.model.BaseResRNNBlock.__init__`.

Usage:
    python -m unittest test/lmp/model/_base_res_rnn_block/test_init.py
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

from lmp.model import BaseResRNNBlock


class TestInit(unittest.TestCase):
    r"""Test Case for `lmp.model.BaseResRNNBlock.__init__`."""

    def setUp(self):
        r"""Set up hyper parameters and construct BaseResRNNBlock"""
        self.d_hid = 1
        self.dropout = 0.1

        Parameters = (
            (
                ('d_hid', self.d_hid),
                ('dropout', self.dropout),
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
                BaseResRNNBlock(*pos),
                BaseResRNNBlock(**kwargs),
            ]

    def tearDown(self):
        r"""Delete parameters and models."""
        del self.d_hid
        del self.dropout
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseResRNNBlock.__init__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='d_hid',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='dropout',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

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
                BaseResRNNBlock(
                    d_hid=invalid_input,
                    dropout=0.1,
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
                BaseResRNNBlock(
                    d_hid=1,
                    dropout=invalid_input
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

    def test_instance_attribute_rnn_layer(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Instance {} attribute `{}` must include `{}`.'

        rnn_examples = (
            ('input_size', self.d_hid),
            ('hidden_size', self.d_hid),
            ('batch_first', True),
        )

        rnn_layer = torch.nn.RNN(
                input_size=self.d_hid,
                hidden_size=self.d_hid,
                batch_first=True
            )

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'rnn_layer'),
                msg=msg1.format('rnn_layer')
            )
            self.assertIsInstance(
                getattr(model, 'rnn_layer'),
                type(rnn_layer),
                msg=msg2.format('rnn_layer', type(rnn_layer).__name__)
            )
            model_layer = getattr(model, 'rnn_layer')
            for rnn_attr, rnn_attr_val in rnn_examples:
                self.assertEqual(
                    getattr(model_layer, rnn_attr),
                    getattr(rnn_layer, rnn_attr),
                    msg=msg3.format('rnn', rnn_attr, rnn_attr_val)
                )

    def test_instance_attribute_dropout(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Instance {} attribute `{}` must need `{}`.'


        dropout_examples = (
            ('p', self.dropout),
        )

        dropout = torch.nn.Dropout(self.dropout)

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'dropout'),
                msg=msg1.format('dropout')
            )
            self.assertIsInstance(
                getattr(model, 'dropout'),
                type(dropout),
                msg=msg2.format('dropout', type(dropout).__name__)
            )
            dropout_layer = getattr(model, 'dropout')
            for dropout_attr, dropout_attr_val in dropout_examples:
                self.assertEqual(
                    getattr(dropout_layer, dropout_attr),
                    getattr(dropout, dropout_attr),
                    msg=msg3.format('dropout', dropout_attr, dropout_attr_val)
                )

    def test_instance_attribute_act_fn(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg4 = 'Inconsitent activation function.'

        act_fn = torch.nn.ReLU()

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'act_fn'),
                msg=msg1.format('act_fn')
            )
            self.assertIsInstance(
                getattr(model, 'act_fn'),
                type(act_fn),
                msg=msg2.format('act_fn', type(act_fn).__name__)
            )
            model_act_fn = getattr(model, 'act_fn')
            self.assertEqual(
                model_act_fn(torch.tensor([10])),
                act_fn(torch.tensor([10])),
                msg=msg4
            )


if __name__ == '__main__':
    unittest.main()
