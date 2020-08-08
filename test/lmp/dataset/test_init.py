r"""Test `lmp.dataset.BaseDataset.__init__`.

Usage:
    python -m unittest test/lmp/dataset/test_init.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

from typing import Iterable

# self-made modules

from lmp.dataset import BaseDataset

class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.__init__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseDataset.__init__),
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
                        annotation=Iterable[str],
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequences(self):
        r"""Raise when `batch_sequences` is invalid."""
        msg1 = 'Must raise `TypeError` when `batch_sequences` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ..., (0.0,), (1.0,), (math.nan,), (-math.nan,),
            (math.inf,), (-math.inf,), (0j,), (1j,), (object(),),
            (lambda x: x,), (type,), (None,), (NotImplemented,), (...,),
            [0.0], [1.0], [math.nan], [-math.nan], [math.inf], [-math.inf],
            [0j], [1j], [object()], [lambda x: x], [type], [None],
            [NotImplemented], [...], {0.0}, {1.0}, {math.nan}, {-math.nan},
            {math.inf}, {-math.inf}, {0j}, {1j}, {object()}, {lambda x: x},
            {type}, {None}, {NotImplemented}, {...}, {0.0: 0}, {1.0: 0},
            {math.nan: 0}, {-math.nan: 0}, {math.inf: 0}, {-math.inf: 0},
            {0j: 0}, {1j: 0}, {object(): 0}, {lambda x: x: 0}, {type: 0},
            {None: 0}, {NotImplemented: 0}, {...: 0},
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                BaseDataset(batch_sequences=invalid_input)

                if isinstance(ctx_man.exception, TypeError):
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_sequences` must be instance of `Iterable[str]`.',
                        msg=msg2
                    )

    def test_instance_attributes(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Instance attribute `{}` must be `{}`.'
        examples = (
            (
                'batch_sequences',
                ['Hello world!', 'Hello apple.', 'Hello gogoro!']
            ),
        )

        for attr, attr_val in examples:
            dataset = BaseDataset(batch_sequences=attr_val)
            self.assertTrue(
                hasattr(dataset, attr),
                msg=msg1
            )
            self.assertIsInstance(
                getattr(dataset,attr),
                type(attr_val),
                msg=msg2.format(attr, type(attr_val).__name__)
            )
            self.assertEqual(
                getattr(dataset, attr),
                attr_val,
                msg=msg3.format(attr, attr_val)
            )

if __name__ == '__main__':
    unittest.main()