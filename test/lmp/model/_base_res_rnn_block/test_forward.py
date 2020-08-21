r"""Test `lmp.model.BaseResRNNBlock.forward`.

Usage:
    python -m unittest test.lmp.model._base_res_rnn_block.test_forward
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


class TestForward(unittest.TestCase):
    r"""Test case for `lmp.model.BaseResRNNBlock.forward`."""

    @classmethod
    def setUpClass(cls):
        cls.batch_range = [1, 2]
        cls.d_hid_range = [1, 10]
        cls.dropout_range = [0.0, 0.1, 0.5, 1.0]
        cls.sequence_range = list(range(1, 5))

    @classmethod
    def tearDownClass(cls):
        del cls.batch_range
        del cls.d_hid_range
        del cls.dropout_range
        del cls.sequence_range
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
            inspect.signature(BaseResRNNBlock.forward),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='x',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.Tensor,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=torch.Tensor
            ),
            msg=msg
        )

    def test_invalid_input_x(self):
        r"""Raise `AttributeError` when input `x` is invalid."""
        msg = 'Must raise `AttributeError` when input `x` is invalid.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            for model_obj in self.model_objs:
                model = model_obj['model']
                with self.assertRaises(AttributeError, msg=msg):
                    model.forward(x=invalid_input)

    def test_return_type(self):
        r"""Return `Tensor`."""
        msg = 'Must return `Tensor`.'

        examples = (
            (
                torch.rand(batch_size, sequence_len, model_obj['d_hid']),
                model_obj['model']
            )
            for batch_size in self.__class__.batch_range
            for sequence_len in self.__class__.sequence_range
            for model_obj in self.model_objs
        )

        for x, model in examples:
            self.assertIsInstance(model(x), torch.Tensor, msg=msg)

    def test_return_size(self):
        r"""Test return size"""
        msg = 'Return size must be {}.'
        examples = (
            (
                torch.rand(batch_size, sequence_len, model_obj['d_hid']),
                model_obj['model']
            )
            for batch_size in self.__class__.batch_range
            for sequence_len in self.__class__.sequence_range
            for model_obj in self.model_objs
        )

        for x, model in examples:
            logits = model(x)
            for s1, s2 in zip(x.size(), logits.size()):
                self.assertEqual(s1, s2, msg=msg)


if __name__ == '__main__':
    unittest.main()
