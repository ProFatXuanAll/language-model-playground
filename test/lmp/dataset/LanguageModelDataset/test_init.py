r"""Test `lmp.dataset.LanguageModelDataset.__init__`.

Usage:
    python -m unittest test/lmp/dataset/LanguageModelDataset/test_init.py
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

from lmp.dataset._language_model_dataset import LanguageModelDataset

class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.LanguageModelDataset.__init__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(LanguageModelDataset.__init__),
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
        r"""Raise `TypeError` when input `batch_sequences` is invalid."""
        msg1 = (
            'Must raise `TypeError` when input `batch_sequences` is invalid.'
        )
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
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                LanguageModelDataset(batch_sequences=invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`batch_sequences` must be an instance of `Iterable[str]`.',
                msg=msg2
            )

    def test_instance_attributes(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        examples = (('batch_sequences', list),)

        for attr, attr_type in examples:
            dataset = LanguageModelDataset(batch_sequences=[])
            self.assertTrue(
                hasattr(dataset, attr),
                msg=msg1.format(attr)
            )
            self.assertIsInstance(
                getattr(dataset, attr),
                attr_type,
                msg=msg2.format(attr, attr_type.__name__)
            )


if __name__ == '__main__':
    unittest.main()
