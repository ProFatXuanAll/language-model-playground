r"""Test `lmp.model.ResLSTMBlock.forward`.

Usage:
    python -m unittest test/lmp/model/_res_lstm_block/test_forward.py
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

from lmp.model import ResLSTMBlock


class TestInit(unittest.TestCase):
    r"""Test Case for `lmp.model.ResLSTMBlock.forward`."""

    def setUp(self):
        r"""Set up hyper parameters and construct ResLSTMBlock"""
        self.d_hid = 2
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
                ResLSTMBlock(*pos),
                ResLSTMBlock(**kwargs),
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
            inspect.signature(ResLSTMBlock.forward),
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
        r"""Raise when `x` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `x` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        error_msg = "'{}' object has no attribute 'size'"
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            for model in self.models:
                with self.assertRaises(AttributeError, msg=msg1) as ctx_man:
                    model.forward(x=invalid_input)

                if isinstance(ctx_man.exception, AttributeError):
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        error_msg.format(type(invalid_input).__name__),
                        msg=msg2
                    )

    def test_return_type(self):
        r"""Return `Tensor`."""
        msg = 'Must return `Tensor`.'

        examples = (
            torch.rand(5, 3, self.d_hid),
            torch.rand(3, 5, self.d_hid),
        )

        for x in examples:
            for model in self.models:
                pred_y= model(x)
                self.assertIsInstance(pred_y, torch.Tensor, msg=msg)

if __name__ == '__main__':
    unittest.main()
