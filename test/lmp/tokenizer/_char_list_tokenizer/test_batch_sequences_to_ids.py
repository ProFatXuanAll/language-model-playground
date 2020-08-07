r"""Test `lmp.tokenizer.CharListTokenizer.batch_sequences_to_tokens`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_list_tokenizer/test_batch_sequences_to_tokens.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import math
import unittest

from typing import Iterable
from typing import List

# self-made modules

from lmp.tokenizer import CharListTokenizer


class TestBatchSequencesToIds(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharListTokenizer.batch_sequences_to_tokens`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharListTokenizer()
        self.uncased_tokenizer = CharListTokenizer(is_uncased=True)
        self.tokenizers = [self.cased_tokenizer, self.uncased_tokenizer]

    def tearDown(self):
        r"""Delete both cased and uncased tokenizer instances."""
        del self.tokenizers
        del self.cased_tokenizer
        del self.uncased_tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(CharListTokenizer.batch_sequences_to_ids),
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
                return_annotation=List[List[int]]
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequences(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False,
            object(), lambda x: x, type, None, 0j, 1j, NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.batch_sequences_to_ids(
                        batch_sequences=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_sequences` must be instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[List[str]]`."""
        msg = 'Must return `List[List[int]]`.'
        examples = (
            [
                'hello',
                'world'
            ],
        )

        for batch_sequences in examples:
            for tokenizer in self.tokenizers:
                batch_token_ids = tokenizer.batch_sequences_to_ids(
                    batch_sequences=batch_sequences
                )
                self.assertIsInstance(batch_token_ids, list, msg=msg)
                for token_ids in batch_token_ids:
                    self.assertIsInstance(token_ids, list, msg=msg)
                    for token_id in token_ids:
                        self.assertIsInstance(token_id, int, msg=msg)

    def test_function(self):
        r"""Test token id look up on batch of sequences."""
        msg = 'Inconsistent error message.'

        examples = (
            (
                [
                    'Hello',
                    'world'
                ],
                [
                    [3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3]
                ]
            ),
            (
                [''],
                [[]]
            ),
        )

        for batch_sequences, ans_batch_token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_sequences_to_ids(
                        batch_sequences=batch_sequences
                    ),
                    ans_batch_token_ids,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
