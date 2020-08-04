r"""Test `lmp.tokenizer.CharDictTokenizer.batch_encode`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/test_batch_encode.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import gc
import math
import unittest

from typing import List

# self-made modules

from lmp.tokenizer import CharDictTokenizer

class TestBatchEncode(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.batch_encode`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharDictTokenizer()
        self.uncased_tokenizer = CharDictTokenizer(is_uncased=True)
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
            inspect.signature(CharDictTokenizer.batch_encode),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        name='batch_sequences',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=List[str],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=-1
                    )
                ],
                return_annotation=List[List[int]]
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequence(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False, b'',
            (), {}, set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.batch_encode(batch_sequences=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_sequences` must be instance of `List[str]`.',
                    msg=msg2
                )

    def test_invalid_input_max_seq_len(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            math.nan, math.inf, b'', [], (), {},
            set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.batch_encode(
                        batch_sequences=['Hello world', 'Goodbye world'],
                        max_seq_len=invalid_input
                    )

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`max_seq_len` must be instance of `int`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[List[int]]`."""
        msg = 'Must return `List[List[int]]`.'
        examples = (
            [
                'Hello world!',
                'Goodbye world!'
            ],
            [''],
        )

        for batch_sequences in examples:
            for tokenizer in self.tokenizers:
                batch_token_ids = tokenizer.batch_encode(batch_sequences=batch_sequences)
                self.assertIsInstance(batch_token_ids, list, msg=msg)
                for token_ids in batch_token_ids:
                    self.assertIsInstance(token_ids, list, msg=msg)
                    for token_id in token_ids:
                        self.assertIsInstance(token_id, int, msg=msg)

    def test_prefix_postfix(self):
        r"""Return List[List[int]] each tokens_ids must include [BOS] id and
        [EOS] id."""
        msg = (
            'Return result must include [BOS] id and [EOS] id for \
                each tokens_ids.'
        )
        examples= (
            (
                [
                    'Hello',
                    'world'
                ],
                [
                    [0, 3, 3, 3, 3, 3, 1],
                    [0, 3, 3, 3, 3, 3, 1]
                ]
            ),
        )

        for batch_sequences, ans_tokens_ids in examples:
            self.assertEqual(
                self.cased_tokenizer.batch_encode(
                    batch_sequences=batch_sequences
                ),
                ans_tokens_ids,
                msg=msg
            )

    def test_truncate(self):
        r"""Return List[List[int]] must make sure each `List[int]`'s length
        must not exceed `max_seq_len`.
        """
        msg = (
            'Return result must make sure  each `List[int]`\'s length must not \
            exceed `max_seq_len`.'
        )
        examples= (
            (
                [
                    'Hello',
                    'world'
                ],
                [
                    [0, 3, 1],
                    [0, 3, 1]
                ]
            ),
        )

        for batch_sequences, ans_tokens_ids in examples:
            self.assertEqual(
                self.cased_tokenizer.batch_encode(
                    batch_sequences=batch_sequences,
                    max_seq_len=3
                ),
                ans_tokens_ids,
                msg=msg
            )

    def test_padding(self):
        r"""Return List[List[int]] must pad each `List[int]` the length to
        `max_seq_len`.
        """
        msg = (
            'Return result must pad each `List[int]` the length to `max_seq_len`.'
        )
        examples= (
            (
                [
                    'Hello',
                    'world'
                ],
                [
                    [0, 3, 3, 3, 3, 3, 1, 2, 2, 2],
                    [0, 3, 3, 3, 3, 3, 1, 2, 2, 2]
                ]
            ),
        )

        for batch_sequences, ans_tokens_ids in examples:
            self.assertEqual(
                self.cased_tokenizer.batch_encode(
                    batch_sequences=batch_sequences,
                    max_seq_len=10
                ),
                ans_tokens_ids,
                msg=msg
            )

if __name__ == '__main__':
    unittest.main()
