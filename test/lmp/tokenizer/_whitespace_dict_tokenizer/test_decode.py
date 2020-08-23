r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.decode`.

Usage:
    python -m unittest \
        test.lmp.tokenizer._whitespace_dict_tokenizer.test_decode
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

# self-made modules

from lmp.tokenizer import WhitespaceDictTokenizer


class TestDecode(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceDictTokenizer.decode`."""

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
            inspect.signature(WhitespaceDictTokenizer.decode),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        name='token_ids',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[int],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='remove_special_tokens',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=False
                    )
                ],
                return_annotation=str
            ),
            msg=msg
        )

    def test_invalid_input_token_ids(self):
        r"""Raise `TypeError` when input `token_ids` is invalid."""
        msg1 = 'Must raise `TypeError` when input `token_ids` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ..., [0.0], [1.0], [math.nan], [-math.nan],
            [math.inf], [-math.inf], [0j], [1j], [''], [b''], [()], [[]], [{}],
            [set()], [object()], [lambda x: x], [type], [None],
            [NotImplemented], [...], [0, 0.0], [0, 1.0], [0, math.nan],
            [0, -math.nan], [0, math.inf], [0, -math.inf], [0, 0j], [0, 1j],
            [0, ''], [0, b''], [0, ()], [0, []], [0, {}], [0, set()],
            [0, object()], [0, lambda x: x], [0, type], [0, None],
            [0, NotImplemented], [0, ...],
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.decode(token_ids=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`token_ids` must be an instance of `Iterable[int]`.',
                    msg=msg2
                )

    def test_invalid_input_remove_special_tokens(self):
        r"""Raise `TypeError` when input `remove_special_tokens` is invalid."""
        msg1 = (
            'Must raise `TypeError` when input `remove_special_tokens` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', (), [], {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.decode(
                        token_ids=[],
                        remove_special_tokens=invalid_input
                    )

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`remove_special_tokens` must be an instance of `bool`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
        examples = (
            [0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0],
            [],
        )

        for token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertIsInstance(
                    tokenizer.decode(token_ids=token_ids),
                    str,
                    msg=msg
                )

    def test_remove_special_tokens(self):
        r"""Remove special tokens."""
        msg = 'Must remove special tokens.'
        examples = (
            (
                False,
                [0, 4, 7, 5, 1, 2],
                '[bos] Hello World ! [eos] [pad]',
                '[bos] hello world ! [eos] [pad]',
            ),
            (
                False,
                [0, 8, 9, 10, 3, 1, 2, 2],
                '[bos] I am a [unk] [eos] [pad] [pad]',
                '[bos] i am a [unk] [eos] [pad] [pad]',
            ),
            (
                False,
                [0, 3, 6, 11, 1],
                '[bos] [unk] legend . [eos]',
                '[bos] [unk] legend . [eos]',
            ),
            (
                True,
                [0, 4, 7, 5, 1, 2],
                'Hello World !',
                'hello world !',
            ),
            (
                True,
                [0, 8, 9, 10, 3, 1, 2, 2],
                'I am a [unk]',
                'i am a [unk]',
            ),
            (
                True,
                [0, 3, 6, 11, 1],
                '[unk] legend .',
                '[unk] legend .',
            ),
        )

        for (
                remove_special_tokens,
                token_ids,
                cased_sequence,
                uncased_sequence
        ) in examples:
            self.assertEqual(
                self.cased_tokenizer.decode(
                    token_ids=token_ids,
                    remove_special_tokens=remove_special_tokens
                ),
                cased_sequence,
                msg=msg
            )
            self.assertEqual(
                self.uncased_tokenizer.decode(
                    token_ids=token_ids,
                    remove_special_tokens=remove_special_tokens
                ),
                uncased_sequence,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
