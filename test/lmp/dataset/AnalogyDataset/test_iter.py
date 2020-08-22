r"""Test `lmp.dataset.AnalogyDataset.__iter__`.

Usage:
    python -m unittest test.lmp.dataset.AnalogyDataset.test_iter
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

from typing import Generator
from typing import Iterable
from typing import List

# self-made modules

from lmp.dataset._analogy_dataset import AnalogyDataset


class TestIter(unittest.TestCase):
    r"""Test case for `lmp.dataset.AnalogyDataset.__iter__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(AnalogyDataset.__iter__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=Generator[List[str], None, None]
            ),
            msg=msg
        )

    def test_yield_value(self):
        r"""Is an iterable which yield analogy test data in order."""
        msg = 'Must be an iterable which yield analogy test data in order.'
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
        )

        for example in examples:
            dataset = AnalogyDataset(samples=example)
            self.assertIsInstance(dataset, Iterable, msg=msg)

            for ans_sequence, sequence in zip(example, dataset):
                self.assertIsInstance(sequence, Iterable, msg=msg)
                for element in sequence:
                    self.assertIsInstance(element, str, msg=msg)
                self.assertEqual(sequence, ans_sequence, msg=msg)


if __name__ == '__main__':
    unittest.main()
