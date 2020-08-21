r"""Test `lmp.dataset.BaseDataset.__getitem__`.

Usage:
    python -m unittest test.lmp.dataset.test_getitem
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


class TestGetItem(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.__getitem__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseDataset.__getitem__),
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
                return_annotation=str
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
                BaseDataset([])[invalid_input]

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`index` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertIsInstance(ctx_man.exception, IndexError)

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
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
            dataset = BaseDataset(batch_sequences=batch_sequences)
            for i in range(len(dataset)):
                self.assertIsInstance(dataset[i], str, msg=msg)

    def test_return_value(self):
        r"""Sample single sequence using index."""
        msg = 'Must sample single sequence using index.'
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
            dataset = BaseDataset(batch_sequences=batch_sequences)
            for i in range(len(dataset)):
                self.assertEqual(dataset[i], batch_sequences[i], msg=msg)


if __name__ == '__main__':
    unittest.main()
