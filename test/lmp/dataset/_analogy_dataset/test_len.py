r"""Test `lmp.dataset.AnalogyDataset.__len__`.

Usage:
    python -m unittest test.lmp.dataset.AnalogyDataset.test_len
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

# self-made modules

from lmp.dataset._analogy_dataset import AnalogyDataset


class TestLen(unittest.TestCase):
    r"""Test case for `lmp.dataset.AnalogyDataset.__len__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(AnalogyDataset.__len__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=int
            ),
            msg=msg
        )

    def test_return_type(self):
        r"""Return `int`."""
        msg = 'Must return `int`.'
        examples = (
            [
                [
                    'Taiwan',
                    'Taipei',
                    'Japan',
                    'Tokyo',
                    'capital',
                ],
                [
                    'write',
                    'writes',
                    'sad',
                    'sads',
                    'grammer',
                ]
            ],
            [
                [
                    'Taiwan',
                    'Taipei',
                    'Japan',
                    'Tokyo',
                    'capital',
                ],
            ],
            [],
        )

        for samples in examples:
            self.assertIsInstance(
                len(AnalogyDataset(samples=samples)),
                int,
                msg=msg
            )

    def test_return_dataset_size(self):
        r"""Return dataset size."""
        msg = 'Must return dataset size.'
        examples = (
            (
                [
                    [
                        'Taiwan',
                        'Taipei',
                        'Japan',
                        'Tokyo',
                        'capital',
                    ],
                    [
                        'write',
                        'writes',
                        'sad',
                        'sads',
                        'grammer',
                    ]
                ],
                2,
            ),
            (
                [
                    [
                        'Taiwan',
                        'Taipei',
                        'Japan',
                        'Tokyo',
                        'capital',
                    ],
                ],
                1,
            ),
            (
                [],
                0,
            ),
        )

        for samples, size in examples:
            self.assertEqual(
                len(AnalogyDataset(samples=samples)),
                size,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
