r"""Test `lmp.dataset.BaseDataset.__len__`.

Usage:
    python -m unittest test/lmp/dataset/test_len.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

# self-made modules

from lmp.dataset import BaseDataset


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.__len`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseDataset.__len__),
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
                'Kimura lock.',
                'Superman punch.'
                'Close Guard.'
            ],
        )

        for batch_sequences in examples:
            dataset = BaseDataset(batch_sequences=batch_sequences)
            self.assertIsInstance(
                len(dataset),
                int,
                msg=msg
            )

    def test_return_value(self):
        r"""Return dataset size."""
        msg = 'Inconsistent error message.'
        examples = (
            ['Hello world!', 'Hello apple.', 'Hello gogoro!'],
            ['Goodbye world!', 'Goodbye apple.'],
            ['Arm bar.', 'Short punch.', 'Right hook.', 'Front kick.'],
        )

        for batch_sequences in examples:
            dataset = BaseDataset(batch_sequences=batch_sequences)
            self.assertEqual(
                len(dataset),
                len(batch_sequences),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
