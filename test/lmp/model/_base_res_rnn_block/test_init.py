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
    r"""Test case for `lmp.model.BaseResRNNBlock.__init__`."""

    def setUp(self):
        r"""Set up hyper parameters and construct BaseResRNNBlock"""
        self.d_hid = 10
        self.dropout = 0.1

        self.model_parameters = (
            (
                ('d_hid', self.d_hid),
                ('dropout', self.dropout),
            ),
        )

    def tearDown(self):
        r"""Delete parameters and models."""
        del self.d_hid
        del self.dropout
        del self.model_parameters
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
        msg1 = 'Must raise `TypeError` when `d_hid` is invalid.'
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
                    '`d_hid` must be an instance of `int`.',
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
        msg1 = 'Must raise `TypeError` when `dropout` is invalid.'
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
                    '`dropout` must be an instance of `float`.',
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
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_hid),
            torch.rand(10, 20, self.d_hid),
        )

        rnn_layer = torch.nn.RNN(
            input_size=self.d_hid,
            hidden_size=self.d_hid,
            batch_first=True
        )

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                BaseResRNNBlock(*pos),
                BaseResRNNBlock(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'rnn_layer'),
                msg=msg1.format('rnn_layer')
            )
            self.assertIsInstance(
                model.rnn_layer,
                type(rnn_layer),
                msg=msg2.format('rnn_layer', type(rnn_layer).__name__)
            )
            for x in examples:
                ht, _ = model.rnn_layer(x)
                ans_out, _ = rnn_layer(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )

    def test_instance_attribute_dropout(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_hid),
            torch.rand(10, 20, self.d_hid),
        )

        dropout = torch.nn.Dropout(self.dropout)

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                BaseResRNNBlock(*pos),
                BaseResRNNBlock(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'dropout'),
                msg=msg1.format('dropout')
            )
            self.assertIsInstance(
                model.dropout,
                type(dropout),
                msg=msg2.format('dropout', type(dropout).__name__)
            )
            for x in examples:
                ht = model.dropout(x)
                ans_out = dropout(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )

    def test_instance_attribute_act_fn(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_hid),
            torch.rand(10, 20, self.d_hid),
        )

        act_fn = torch.nn.ReLU()

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                BaseResRNNBlock(*pos),
                BaseResRNNBlock(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'act_fn'),
                msg=msg1.format('act_fn')
            )
            self.assertIsInstance(
                model.act_fn,
                type(act_fn),
                msg=msg2.format('act_fn', type(act_fn).__name__)
            )
            for x in examples:
                ht = model.act_fn(x)
                ans_out = act_fn(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )


if __name__ == '__main__':
    unittest.main()
