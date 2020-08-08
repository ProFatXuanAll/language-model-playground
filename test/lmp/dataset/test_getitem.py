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
        r"""Raise when `index` is invalid."""
        msg1 = 'Must raise `TypeError` when `index` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                BaseDataset(['a', 'b'])[invalid_input]

            self.assertEqual(
                ctx_man.exception.args[0],
                '`index` must be instance of `int`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
        examples = (
            [
                'Kimura lock.',
                'Superman punch.'
                'Close Guard.'
            ],
        )

        for batch_sequences in examples:
            for sequence in BaseDataset(batch_sequences=batch_sequences):
                self.assertIsInstance(
                    sequence,
                    str,
                    msg=msg
                )

    def test_return_value(self):
        r"""Return Sample single sequence using index."""
        msg = 'Inconsistent error message.'
        examples = (
                ['Hello world!', 'Hello apple.', 'Hello gogoro!'],
                ['Goodbye world!', 'Goodbye apple.'],
                ['Arm bar.', 'Short punch.', 'Right hook.', 'Front kick.'],
        )

        for batch_sequences in examples:
            dataset = BaseDataset(batch_sequences=batch_sequences)
            for i, ans_sequence in enumerate(batch_sequences):
                self.assertEqual(
                    dataset[i],
                    ans_sequence,
                    msg=msg
                )

if __name__ == '__main__':
    unittest.main()