r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.tokenize`.

Usage:
    python -m unittest \
        test.lmp.tokenizer._whitespace_dict_tokenizer.test_tokenize
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import math
import unicodedata
import unittest

from typing import List

# self-made modules

from lmp.tokenizer import WhitespaceDictTokenizer


class TestTokenize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceDictTokenizer.tokenize`."""

    @classmethod
    def setUpClass(cls):
        cls.vocab_source = [
            'Hello World !',
            'I am a legend .',
            'Hello legend !',
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.vocab_source
        gc.collect()

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = WhitespaceDictTokenizer()
        self.cased_tokenizer.build_vocab(self.__class__.vocab_source)
        self.uncased_tokenizer = WhitespaceDictTokenizer(is_uncased=True)
        self.uncased_tokenizer.build_vocab(self.__class__.vocab_source)
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
            inspect.signature(WhitespaceDictTokenizer.tokenize),
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

    def test_invalid_input_sequence(self):
        r"""Raise `TypeError` when input `sequence` is invalid."""
        msg1 = 'Must raise `TypeError` when input `sequence` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, b'', 0j, 1j, [], (), {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                    tokenizer.tokenize(invalid_input)

                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`sequence` must be an instance of `str`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        examples = (
            'Hello World !',
            'H',
            '',
        )

        for sequence in examples:
            for tokenizer in self.tokenizers:
                tokens = tokenizer.tokenize(sequence)
                self.assertIsInstance(tokens, list, msg=msg)
                for token in tokens:
                    self.assertIsInstance(token, str, msg=msg)

    def test_normalize(self):
        r"""Return sequence is normalized."""
        msg = 'Return sequence must be normalized.'
        examples = (
            (
                ' HeLlO WoRlD !',
                ['HeLlO', 'WoRlD', '!'],
                ['hello', 'world', '!'],
            ),
            (
                'HeLlO WoRlD ! ',
                ['HeLlO', 'WoRlD', '!'],
                ['hello', 'world', '!'],
            ),
            (
                '  HeLlO  WoRlD !  ',
                ['HeLlO', 'WoRlD', '!'],
                ['hello', 'world', '!'],
            ),
            (
                '０',
                ['0'],
                ['0'],
            ),
            (
                'é',
                [unicodedata.normalize('NFKC', 'é')],
                [unicodedata.normalize('NFKC', 'é')],
            ),
            (
                '０ é',
                [
                    unicodedata.normalize('NFKC', '0'),
                    unicodedata.normalize('NFKC', 'é'),
                ],
                [
                    unicodedata.normalize('NFKC', '0'),
                    unicodedata.normalize('NFKC', 'é'),
                ],
            ),
            (
                '',
                [],
                [],
            ),
        )

        for sequence, cased_tokens, uncased_tokens in examples:
            self.assertEqual(
                self.cased_tokenizer.tokenize(sequence),
                cased_tokens,
                msg=msg
            )
            self.assertEqual(
                self.uncased_tokenizer.tokenize(sequence),
                uncased_tokens,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
