r"""Test `lmp.dataset.AnalogyDataset.__getitem__`.

Usage:
    python -m unittest test.lmp.dataset.AnalogyDataset.test_getitem
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

from typing import List

# self-made modules

from lmp.dataset._analogy_dataset import AnalogyDataset


class TestGetItem(unittest.TestCase):
    r"""Test case for `lmp.dataset.AnalogyDataset.__getitem__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(AnalogyDataset.__getitem__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='index',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=List[str]
            ),
            msg=msg
        )

    def test_invalid_input_index(self):
        r"""Raise `IndexError` or `TypeError` when `index` is invalid."""
        msg1 = (
            'Must raise `IndexError` or `TypeError` when `index` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (IndexError, TypeError),
                    msg=msg1
            ) as ctx_man:
                AnalogyDataset([])[invalid_input]

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`index` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertIsInstance(ctx_man.exception, IndexError)

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        msg2 = 'Sample must have exactly 5 elements.'
        examples = (
            (
                (
                    ['Taiwan', 'Taipei', 'Japan', 'Tokyo', 'capital'],
                    ['write', 'writes', 'sad', 'sads', 'grammer'],
                ),
                2,
            ),
            (
                (
                    ['write', 'writes', 'sad', 'sads', 'grammer'],
                ),
                1,
            ),
            (
                (),
                0,
            ),
        )

        for samples, size in examples:
            dataset = AnalogyDataset(samples=samples)
            for index in range(size):
                sample = dataset[index]
                self.assertIsInstance(dataset[index], list, msg=msg)
                # Make sure there are only 5 elements in each data.
                self.assertEqual(len(sample), 5, msg=msg2)
                for item in sample:
                    self.assertIsInstance(item, str, msg=msg)

    def test_return_value(self):
        r"""Sample single analogy data using index."""
        msg = 'Must single analogy data using index.'
        examples = (
            (
                (
                    ['Taiwan', 'Taipei', 'Japan', 'Tokyo', 'capital'],
                    ['write', 'writes', 'sad', 'sads', 'grammer'],
                ),
                2,
            ),
            (
                (
                    ['write', 'writes', 'sad', 'sads', 'grammer'],
                ),
                1,
            ),
            (
                (),
                0,
            ),
        )

        for samples, size in examples:
            dataset = AnalogyDataset(samples=samples)
            for index in range(size):
                self.assertEqual(dataset[index], samples[index], msg=msg)


if __name__ == '__main__':
    unittest.main()
