r"""Test `lmp.tokenizer.WhitespaceListTokenizer.detokenize`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_list_tokenizer/test_detokenize.py
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

from typing import Iterable

# self-made modules

from lmp.tokenizer import WhitespaceListTokenizer


class TestDetokenize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceListTokenizer.detokenize`."""

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
        self.cased_tokenizer = WhitespaceListTokenizer()
        self.cased_tokenizer.build_vocab(self.__class__.vocab_source)
        self.uncased_tokenizer = WhitespaceListTokenizer(is_uncased=True)
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
            inspect.signature(WhitespaceListTokenizer.detokenize),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokens',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[str],
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=str
            ),
            msg=msg
        )

    def test_invalid_input_tokens(self):
        r"""Raise `TypeError` when input `tokens` is invalid."""
        msg1 = 'Must raise `TypeError` when input `tokens` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ..., [False], [True], [0], [1], [-1], [0.0], [1.0],
            [math.nan], [-math.nan], [math.inf], [-math.inf], [0j], [1j],
            [b''], [()], [[]], [{}], [set()], [object()], [lambda x: x],
            [type], [None], [NotImplemented], [...], ['', False], ['', True],
            ['', 0], ['', 1], ['', -1], ['', 0.0], ['', 1.0], ['', math.nan],
            ['', -math.nan], ['', math.inf], ['', -math.inf], ['', 0j],
            ['', 1j], ['', b''], ['', ()], ['', []], ['', {}], ['', set()],
            ['', object()], ['', lambda x: x], ['', type], ['', None],
            ['', NotImplemented], ['', ...],
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                    tokenizer.detokenize(tokens=invalid_input)

                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokens` must be an instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
        examples = (
            ('HeLlO', 'WoRlD', '!'),
            (''),
            (),
        )

        for tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertIsInstance(
                    tokenizer.detokenize(tokens),
                    str,
                    msg=msg
                )

    def test_normalize(self):
        r"""Return sequence is normalized."""
        msg = 'Return sequence must be normalized.'
        examples = (
            (
                (' ', 'HeLlO', 'WoRlD', '!'),
                'HeLlO WoRlD !',
                'hello world !',
            ),
            (
                ('HeLlO', 'WoRlD', '!', ' '),
                'HeLlO WoRlD !',
                'hello world !',
            ),
            (
                (' ', ' ', 'HeLlO', ' ', ' ', 'WoRlD', '!', ' ', ' '),
                'HeLlO WoRlD !',
                'hello world !',
            ),
            (
                ('０'),
                '0',
                '0',
            ),
            (
                ('é'),
                unicodedata.normalize('NFKC', 'é'),
                unicodedata.normalize('NFKC', 'é'),
            ),
            (
                ('０', 'é'),
                unicodedata.normalize('NFKC', '0 é'),
                unicodedata.normalize('NFKC', '0 é'),
            ),
            (
                (),
                '',
                '',
            ),
        )

        for tokens, cased_sequence, uncased_sequence in examples:
            self.assertEqual(
                self.cased_tokenizer.detokenize(tokens),
                cased_sequence,
                msg=msg
            )
            self.assertEqual(
                self.uncased_tokenizer.detokenize(tokens),
                uncased_sequence,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
