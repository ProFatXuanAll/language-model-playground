r"""Test `lmp.model.BaseResRNNBlock.__init__`.

Usage:
    python -m unittest test.lmp.model._base_res_rnn_block.test_init
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

    @classmethod
    def setUpClass(cls):
        cls.d_hid_range = [1, 10]
        cls.dropout_range = [0.0, 0.1, 0.5, 1.0]

    @classmethod
    def tearDownClass(cls):
        del cls.d_hid_range
        del cls.dropout_range
        gc.collect()

    def setUp(self):
        r"""Setup hyperparameters and construct `BaseResRNNBlock`."""
        self.model_objs = []
        cls = self.__class__
        for d_hid in cls.d_hid_range:
            for dropout in cls.dropout_range:
                self.model_objs.append({
                    'd_hid': d_hid,
                    'dropout': dropout,
                    'model': BaseResRNNBlock(
                        d_hid=d_hid,
                        dropout=dropout
                    ),
                })

    def tearDown(self):
        r"""Delete model instances."""
        del self.model_objs
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
        r"""Raise exception when input `d_hid` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `d_hid` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', (), [], {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseResRNNBlock(d_hid=invalid_input, dropout=0.0)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_dropout(self):
        r"""Raise exception when input `dropout` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `dropout` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, -1.0, 1.1, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseResRNNBlock(d_hid=1, dropout=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must be an instance of `float`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must range from `0.0` to `1.0`.',
                    msg=msg2
                )

    def test_inherit(self):
        r""""Is subclass of `torch.nn.Module`."""
        msg = 'Must be subclass of `torch.nn.Module`.'

        for model_obj in self.model_objs:
            self.assertIsInstance(model_obj['model'], torch.nn.Module, msg=msg)

    def test_instance_attributes(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        examples = (
            ('rnn_layer', torch.nn.RNN),
            ('dropout', torch.nn.Dropout),
            ('act_fn', torch.nn.ReLU),
        )

        for attr, attr_type in examples:
            for model_obj in self.model_objs:
                model = model_obj['model']
                self.assertTrue(hasattr(model, attr), msg=msg1.format(attr))
                self.assertIsInstance(
                    getattr(model, attr),
                    attr_type,
                    msg=msg2.format(attr, attr_type.__name__)
                )

    def test_residual_rnn_layer(self):
        r"""Declare correct residual RNN block with dropout."""
        msg = 'Must declare correct residual RNN block with dropout.'
        examples = (
            (
                model_obj['model'],
                model_obj['d_hid'],
                model_obj['dropout'],
            )
            for model_obj in self.model_objs
        )

        for model, d_hid, dropout in examples:
            self.assertIsInstance(model.rnn_layer, torch.nn.RNN, msg=msg)
            self.assertEqual(model.rnn_layer.input_size, d_hid, msg=msg)
            self.assertEqual(model.rnn_layer.hidden_size, d_hid, msg=msg)
            self.assertEqual(model.rnn_layer.num_layers, 1, msg=msg)
            self.assertTrue(model.rnn_layer.batch_first, msg=msg)
            self.assertIsInstance(model.dropout, torch.nn.Dropout, msg=msg)
            self.assertEqual(model.dropout.p, dropout, msg=msg)
            self.assertIsInstance(model.act_fn, torch.nn.ReLU, msg=msg)


if __name__ == '__main__':
    unittest.main()
