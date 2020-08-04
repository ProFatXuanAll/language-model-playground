r"""Test `lmp.tokenizer.WhitespaceListTokenizer.tokenize`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_list_tokenizer/test_tokenize.py
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


class TestTokenize(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.WhitespaceListTokenizer.tokenize`."""

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
            inspect.signature(WhitespaceListTokenizer.tokenize),
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
                return_annotation=List[str]
            ),
            msg=msg
        )

    def test_invalid_input(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False, b'',
            [], (), {}, set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                    tokenizer.tokenize(invalid_input)

                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`sequence` must be instance of `str`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        examples = (
            'Hello world!',
            '',
        )

        for sequence in examples:
            for tokenizer in self.tokenizers:
                tokens = tokenizer.tokenize(sequence)
                self.assertIsInstance(tokens, list, msg=msg)
                for token in tokens:
                    self.assertIsInstance(token, str, msg=msg)

    def test_unicode_normalize(self):
        r"""Return NFKC normalized characters."""
        msg = 'Must return NFKC normalized characters.'
        examples = (
            ('０', ['0']),
            ('é', ['é']),
            ('０ é', ['0', 'é']),
        )

        for sequence, ans_tokens in examples:
            for tokenizer in self.tokenizers:
                out_tokens = tokenizer.tokenize(sequence)
                self.assertEqual(out_tokens, ans_tokens, msg=msg)
                for out_token in out_tokens:
                    self.assertEqual(len(out_token), 1, msg=msg)

    def test_cased_sensitive(self):
        r"""Return cased sensitive characters when `is_uncased=False`."""
        msg = (
            'Return result must be case-sensitive when construct with '
            '`is_uncased=False`.'
        )
        examples = (
            (
                'HeLlO WoRlD!',
                ['HeLlO', 'WoRlD!']
            ),
            (
                'HELLO WORLD!',
                ['HELLO', 'WORLD!']
            ),
            (
                'hello world!',
                ['hello', 'world!']
            ),
            (
                'H',
                ['H']
            ),
            (
                'h',
                ['h']
            ),
        )

        for sequence, ans_tokens in examples:
            self.assertEqual(
                self.cased_tokenizer.tokenize(sequence),
                ans_tokens,
                msg=msg
            )

    def test_cased_insensitive(self):
        r"""Return cased insensitive characters when `is_uncased=True`."""
        msg = (
            'Return result must be case-insensitive when construct with '
            '`is_uncased=True`.'
        )
        examples = (
            (
                'HeLlO WoRlD!',
                ['hello', 'world!']
            ),
            (
                'HELLO WORLD!',
                ['hello', 'world!']
            ),
            (
                'hello world!',
                ['hello', 'world!']
            ),
            (
                'H',
                ['h']
            ),
            (
                'h',
                ['h']
            ),
        )

        for sequence, ans_tokens in examples:
            self.assertEqual(
                self.uncased_tokenizer.tokenize(sequence),
                ans_tokens,
                msg=msg
            )

    def test_whitespace_strip(self):
        r"""Strip input sequence."""
        msg = (
            'Input sequence must strip both leading and trailing whitespace '
            'characters.'
        )
        examples = (
            (
                '  hello world!  ',
                ['hello', 'world!']
            ),
            (
                '  hello world!',
                ['hello', 'world!']
            ),
            (
                'hello world!  ',
                ['hello', 'world!']
            ),
            (
                '\nhello world!\n',
                ['hello', 'world!']
            ),
            (
                ' ',
                []
            ),
            (
                '',
                []
            ),
        )

        for sequence, ans_tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.tokenize(sequence),
                    ans_tokens,
                    msg=msg
                )

    def test_whitespace_collapse(self):
        r"""Collapse whitespace characters."""
        msg = (
            'Input sequence must convert consecutive whitespace characters '
            'into single whitespace character.'
        )
        examples = (
            (
                'hello  world  !',
                ['hello', 'world', '!']
            ),
            (
                'hello   world  !',
                ['hello', 'world', '!']
            ),
            (
                'hello  world   !',
                ['hello', 'world', '!']
            ),
            (
                'hello  world\n\n!',
                ['hello', 'world', '!']
            ),
        )

        for sequence, ans_tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.tokenize(sequence),
                    ans_tokens,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
