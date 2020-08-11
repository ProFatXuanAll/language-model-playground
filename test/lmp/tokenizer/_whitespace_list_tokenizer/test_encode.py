r"""Test `lmp.tokenizer.WhitespaceListTokenizer.encode`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_list_tokenizer/test_encode.py
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

from lmp.tokenizer import WhitespaceListTokenizer


class TestEncode(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceListTokenizer.encode`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = WhitespaceListTokenizer()
        self.uncased_tokenizer = WhitespaceListTokenizer(is_uncased=True)
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
            inspect.signature(WhitespaceListTokenizer.encode),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        name='sequence',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=-1
                    )
                ],
                return_annotation=List[int]
            ),
            msg=msg
        )

    def test_invalid_input_sequence(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False, b'', 0j, 1j,
            [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.encode(sequence=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`sequence` must be an instance of `str`.',
                    msg=msg2
                )

    def test_invalid_input_max_seq_len(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            math.nan, math.inf, b'', [], (), {}, 0j, 1j, NotImplemented, ...,
            set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.encode(sequence='Hello world',
                                     max_seq_len=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`max_seq_len` must be an instance of `int`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[int]`."""
        msg = 'Must return `List[int]`.'
        examples = (
            'Hello world!',
            '',
        )

        for sequence in examples:
            for tokenizer in self.tokenizers:
                token_ids = tokenizer.encode(sequence=sequence)
                self.assertIsInstance(token_ids, list, msg=msg)
                for token_id in token_ids:
                    self.assertIsInstance(token_id, int, msg=msg)

    def test_prefix_postfix(self):
        r"""Return List[int] must include [BOS] id and [EOS] id"""
        msg = (
            'Return result must include [BOS] id and [EOS] id.'
        )
        examples = (
            (
                'Hello',
                [0, 3, 1]
            ),
        )

        for sequence, ans_token_ids in examples:
            self.assertEqual(
                self.cased_tokenizer.encode(sequence),
                ans_token_ids,
                msg=msg
            )

    def test_truncate(self):
        r"""Return List[int]'s length must not exceed `max_seq_len`."""
        msg = (
            'Return result\'s length must not exceed `max_seq_len`.'
        )
        examples = (
            (
                'Hello world! python',
                [0, 3, 3, 3, 1, 2, 2, 2, 2, 2]
            ),
        )

        for sequence, ans_token_ids in examples:
            self.assertEqual(
                self.cased_tokenizer.encode(sequence=sequence, max_seq_len=10),
                ans_token_ids,
                msg=msg
            )

    def test_padding(self):
        r"""Return List[int] must pad the length to `max_seq_len`."""
        msg = (
            'Return result must pad the length to `max_seq_len`.'
        )
        examples = (
            (
                'Hello',
                [0, 3, 1, 2, 2, 2, 2, 2, 2, 2]
            ),
        )

        for sequence, ans_token_ids in examples:
            self.assertEqual(
                self.cased_tokenizer.encode(sequence=sequence, max_seq_len=10),
                ans_token_ids,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
