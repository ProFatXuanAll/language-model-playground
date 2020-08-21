r"""Test `lmp.dataset.BaseDataset.__len__`.

Usage:
    python -m unittest test.lmp.dataset.test_len
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

# self-made modules

from lmp.dataset import BaseDataset


class TestLen(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.__len__`."""

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
                'Hello',
                'World',
                'Hello World',
            ],
            [
                'Mario use Kimura Lock on Luigi, and Luigi tap out.',
                'Mario use Superman Punch.',
                'Luigi get TKO.',
                'Toad and Toadette are fightting over mushroom (weed).',
            ],
            [''],
            [],
        )

        for batch_sequences in examples:
            self.assertIsInstance(
                len(BaseDataset(batch_sequences=batch_sequences)),
                int,
                msg=msg
            )

    def test_return_dataset_size(self):
        r"""Return dataset size."""
        msg = 'Must return dataset size.'
        examples = (
            (
                [
                    'Hello',
                    'World',
                    'Hello World',
                ],
                3,
            ),
            (
                [
                    'Mario use Kimura Lock on Luigi, and Luigi tap out.',
                    'Mario use Superman Punch.',
                    'Luigi get TKO.',
                    'Toad and Toadette are fightting over mushroom (weed).',
                ],
                4,
            ),
            (
                [''],
                1
            ),
            (
                [],
                0
            ),
        )

        for batch_sequences, dataset_size in examples:
            self.assertEqual(
                len(BaseDataset(batch_sequences=batch_sequences)),
                dataset_size,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
