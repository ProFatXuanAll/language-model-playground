r"""Test `lmp.dataset.BaseDataset.__iter__`.

Usage:
    python -m unittest test/lmp/dataset/test_iter.py
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
from typing import Generator

# self-made modules

from lmp.dataset import BaseDataset


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.__iter__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseDataset.__iter__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=Generator[str, None, None]
            ),
            msg=msg
        )

    def test_yield_value(self):
        r"""Is an iterable which yield attributes in order."""
        msg = 'Must be an iterable which yield attributes in order.'
        examples = (
            (
                ['Hello world!', 'Hello apple.', 'Hello gogoro!'],
                ['Hello world!', 'Hello apple.', 'Hello gogoro!'],
            ),
            (
                ['Goodbye world!', 'Goodbye apple.', 'Goodbye gogoro!'],
                ['Goodbye world!', 'Goodbye apple.', 'Goodbye gogoro!'],
            ),
        )

        for batch_sequences, ans_batch_sequences in examples:
            dataset = BaseDataset(batch_sequences=batch_sequences)
            self.assertIsInstance(dataset, Iterable, msg=msg)

            for i, sequence in enumerate(dataset):
                self.assertIn(sequence, ans_batch_sequences, msg=msg)
                self.assertIsInstance(sequence, str, msg=msg)
                self.assertEqual(
                    sequence,
                    ans_batch_sequences[i],
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
