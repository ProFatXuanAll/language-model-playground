r"""Test `lmp.dataset.AnalogyDataset.__init__`.

Usage:
    python -m unittest test.lmp.dataset.AnalogyDataset.test_init
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
from typing import List

# self-made modules

from lmp.dataset._analogy_dataset import AnalogyDataset


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.AnalogyDataset.__init__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(AnalogyDataset.__init__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='samples',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[Iterable[str]],
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input_samples(self):
        r"""Raise `TypeError` when input `samples` is invalid."""
        msg1 = (
            'Must raise `TypeError` when input `samples` is invalid.'
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
            with self.assertRaises(
                    (TypeError, IndexError),
                    msg=msg1
                ) as ctx_man:
                AnalogyDataset(samples=invalid_input)

            if isinstance(ctx_man.exception, IndexError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    'Every sample must have word_a, word_b, word_c, word_d'
                    ' and categoty.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`samples` must be an instance of `Iterable[Iterable[str]]`.',
                    msg=msg2
                )

    def test_instance_attributes(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        examples = (('samples', list),)

        for attr, attr_type in examples:
            dataset = AnalogyDataset(samples=[
                [
                    'Taiwan',
                    'Taipei',
                    'Japan',
                    'Tokyo',
                    'capital',
                ]
            ])
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
