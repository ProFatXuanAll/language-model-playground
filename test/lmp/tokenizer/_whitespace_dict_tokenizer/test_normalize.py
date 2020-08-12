r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.normalize`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_dict_tokenizer/test_normalize.py
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

# self-made modules

from lmp.tokenizer import WhitespaceDictTokenizer


class TestNormalize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceDictTokenizer.normalize`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = WhitespaceDictTokenizer()
        self.uncased_tokenizer = WhitespaceDictTokenizer(is_uncased=True)
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
            inspect.signature(WhitespaceDictTokenizer.normalize),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='sequence',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=str
            ),
            msg=msg
        )

    def test_invalid_input_sequence(self):
        r"""Raise `TypeError` when input `sequence` is invalid."""
        msg1 = 'Must raise `TypeError` when input `sequence` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.normalize(sequence=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`sequence` must be an instance of `str`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
        examples = (
            'Hello world!',
            'I am a legend.',
            'y = f(x)',
            '',
        )

        for sequence in examples:
            for tokenizer in self.tokenizers:
                self.assertIsInstance(
                    tokenizer.normalize(sequence=sequence),
                    str,
                    msg=msg
                )

    def test_unicode_normalize(self):
        r"""Return NFKC normalized characters."""
        msg = 'Must return NFKC normalized characters.'
        examples = (
            ('０', '0', 1),
            ('é', 'é', 1),
            ('０é', '0é', 2),
        )

        for sequence, normalized_sequence, sequence_len in examples:
            for tokenizer in self.tokenizers:
                out_sequence = tokenizer.normalize(sequence=sequence)
                self.assertEqual(out_sequence, normalized_sequence, msg=msg)
                self.assertEqual(len(out_sequence), sequence_len, msg=msg)

    def test_cased_sensitive(self):
        r"""Return cased sensitive sequence."""
        msg = 'Return sequence must be cased sensitive.'
        examples = (
            ('HeLlO WoRlD!', 'HeLlO WoRlD!', 'hello world!'),
            ('HELLO WORLD!', 'HELLO WORLD!', 'hello world!'),
            ('hello world!', 'hello world!', 'hello world!'),
            ('H', 'H', 'h'),
            ('h', 'h', 'h'),
        )

        for sequence, cased_sequence, uncased_sequence in examples:
            self.assertEqual(
                self.cased_tokenizer.normalize(sequence),
                cased_sequence,
                msg=msg
            )
            self.assertEqual(
                self.uncased_tokenizer.normalize(sequence),
                uncased_sequence,
                msg=msg
            )

    def test_whitespace_strip(self):
        r"""Strip input sequence."""
        msg = 'Must strip both leading and trailing whitespace characters.'
        examples = (
            (' hello world!', 'hello world!'),
            ('hello world! ', 'hello world!'),
            (' hello world! ', 'hello world!'),
            ('  hello world!   ', 'hello world!'),
            ('\nhello world!\n', 'hello world!'),
            (' ', ''),
            ('', ''),
        )

        for sequence, stripped_sequence in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.normalize(sequence),
                    stripped_sequence,
                    msg=msg
                )

    def test_whitespace_collapse(self):
        r"""Collapse whitespace characters."""
        msg = (
            'Must convert consecutive whitespace characters into single '
            'whitespace character.'
        )
        examples = (
            ('hello  world  !', 'hello world !'),
            ('hello   world  !', 'hello world !'),
            ('hello  world   !', 'hello world !'),
            ('hello  world\n\n!', 'hello world !'),
        )

        for sequence, ans_tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.normalize(sequence),
                    ans_tokens,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
