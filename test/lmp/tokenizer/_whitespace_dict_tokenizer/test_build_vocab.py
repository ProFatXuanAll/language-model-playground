r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.build_vocab`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_dict_tokenizer/test_build_vocab.py
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


class TestBuildVocab(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceDictTokenizer.build_vocab`."""

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
            inspect.signature(WhitespaceDictTokenizer.build_vocab),
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
                    inspect.Parameter(
                        name='min_count',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequences(self):
        r"""Raise `TypeError` when input `batch_sequences` is invalid."""
        msg1 = (
            'Must raise `TypeError` when input `batch_sequences` is invalid.'
        )
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
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.build_vocab(batch_sequences=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_sequences` must be an instance of '
                    '`Iterable[str]`.',
                    msg=msg2
                )

    def test_invalid_input_min_count(self):
        r"""Raise `TypeError` when input `min_count` is invalid."""
        msg1 = 'Must raise `TypeError` when input `min_count` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', (), [], {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.build_vocab(
                        batch_sequences=[],
                        min_count=invalid_input
                    )

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`min_count` must be an instance of `int`.',
                    msg=msg2
                )

    def test_cased_sensitive(self):
        r"""Vocabulary must be case sensitive."""
        msg = 'Vocabulary must be case sensitive.'
        examples = (
            (('A B C D', 'a b c d'), 8, 4),
            (('e f g h i', 'E F G H I'), 10, 5),
        )

        sp_tokens_size = len(list(WhitespaceDictTokenizer.special_tokens()))

        for batch_sequences, cased_vocab_size, uncased_vocab_size in examples:
            self.cased_tokenizer.reset_vocab()
            self.cased_tokenizer.build_vocab(batch_sequences=batch_sequences)
            self.assertEqual(
                self.cased_tokenizer.vocab_size,
                cased_vocab_size + sp_tokens_size,
                msg=msg
            )
            self.uncased_tokenizer.reset_vocab()
            self.uncased_tokenizer.build_vocab(batch_sequences=batch_sequences)
            self.assertEqual(
                self.uncased_tokenizer.vocab_size,
                uncased_vocab_size + sp_tokens_size,
                msg=msg
            )

    def test_sort_by_token_frequency_in_descending_order(self):
        r"""Sort vocabulary by token frequency in descending order."""
        msg = (
            'Must sort vocabulary by token frequency in descending order.'
        )
        examples = (
            (
                ('A a A a', 'b B b', 'c C', 'd'),
                ('A', 'a', 'b', 'B', 'c', 'C', 'd'),
                ('a', 'b', 'c', 'd'),
            ),
            (
                ('E e E e E', 'F f F f', 'G g G', 'H h', 'I'),
                ('E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I'),
                ('e', 'f', 'g', 'h', 'i'),
            ),
        )

        for (
                batch_sequences,
                cased_vocab_order,
                uncased_vocab_order
        ) in examples:
            self.cased_tokenizer.reset_vocab()
            self.cased_tokenizer.build_vocab(batch_sequences=batch_sequences)

            for (
                    vocab1,
                    vocab2
            ) in zip(cased_vocab_order[:-1], cased_vocab_order[1:]):
                self.assertLessEqual(
                    self.cased_tokenizer.convert_token_to_id(vocab1),
                    self.cased_tokenizer.convert_token_to_id(vocab2),
                    msg=msg
                )

            self.uncased_tokenizer.reset_vocab()
            self.uncased_tokenizer.build_vocab(batch_sequences=batch_sequences)

            for (
                    vocab1,
                    vocab2
            ) in zip(uncased_vocab_order[:-1], uncased_vocab_order[1:]):
                self.assertLessEqual(
                    self.uncased_tokenizer.convert_token_to_id(vocab1),
                    self.uncased_tokenizer.convert_token_to_id(vocab2),
                    msg=msg
                )

    def test_min_count(self):
        r"""Filter out tokens whose frequency is smaller than `min_count`."""
        msg = (
            'Must filter out tokens whose frequency is smaller than '
            '`min_count`.'
        )
        examples = (
            (
                ('A a A a', 'b B b', 'c C', 'd'),
                ('A', 'a', 'b'),
                ('B', 'c', 'C', 'd'),
                ('a', 'b', 'c'),
                ('d'),
                2,
            ),
            (
                ('E e E e E', 'F f F f', 'G g G', 'H h', 'I'),
                ('E'),
                ('e', 'F', 'f', 'G', 'g', 'H', 'h', 'I'),
                ('e', 'f', 'g'),
                ('h', 'i'),
                3,
            ),
            (
                ('E e E e E', 'F f F f', 'G g G', 'H h', 'I'),
                (),
                ('E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I'),
                (),
                ('e', 'f', 'g', 'h', 'i'),
                10,
            ),
        )

        for (
                batch_sequences,
                cased_known_token,
                cased_unknown_token,
                uncased_known_token,
                uncased_unknown_token,
                min_count
        ) in examples:
            self.cased_tokenizer.reset_vocab()
            self.cased_tokenizer.build_vocab(
                batch_sequences=batch_sequences,
                min_count=min_count
            )

            for token in cased_known_token:
                token_id = self.cased_tokenizer.convert_token_to_id(token)
                self.assertEqual(
                    token,
                    self.cased_tokenizer.convert_id_to_token(token_id),
                    msg=msg
                )

            unk_token_id = self.cased_tokenizer.convert_token_to_id(
                WhitespaceDictTokenizer.unk_token
            )
            for unk_token in cased_unknown_token:
                self.assertEqual(
                    self.cased_tokenizer.convert_token_to_id(unk_token),
                    unk_token_id,
                    msg=msg
                )

            self.uncased_tokenizer.reset_vocab()
            self.uncased_tokenizer.build_vocab(
                batch_sequences=batch_sequences,
                min_count=min_count
            )

            for token in uncased_known_token:
                token_id = self.uncased_tokenizer.convert_token_to_id(token)
                self.assertEqual(
                    token,
                    self.uncased_tokenizer.convert_id_to_token(token_id),
                    msg=msg
                )

            unk_token_id = self.uncased_tokenizer.convert_token_to_id(
                WhitespaceDictTokenizer.unk_token
            )
            for unk_token in uncased_unknown_token:
                self.assertEqual(
                    self.uncased_tokenizer.convert_token_to_id(unk_token),
                    unk_token_id,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
