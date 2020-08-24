r"""Test `lmp.tokenizer.CharListTokenizer.batch_encode`.

Usage:
    python -m unittest \
        test.lmp.tokenizer._char_list_tokenizer.test_batch_encode
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


class TestBatchEncode(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharListTokenizer.batch_encode`."""

    @classmethod
    def setUpClass(cls):
        cls.vocab_source = [
            'Hello World!',
            'I am a legend.',
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.vocab_source
        gc.collect()

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharListTokenizer()
        self.cased_tokenizer.build_vocab(self.__class__.vocab_source)
        self.uncased_tokenizer = CharListTokenizer(is_uncased=True)
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
            inspect.signature(CharListTokenizer.batch_encode),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        name='batch_sequences',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[str],
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
                    tokenizer.batch_encode(batch_sequences=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_sequences` must be an instance of `Iterable[str]`.',
                    msg=msg2)

    def test_invalid_input_max_seq_len(self):
        r"""Raise exception when input `max_seq_len` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `max_seq_len` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -2, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(
                        (TypeError, ValueError),
                        msg=msg1
                ) as cxt_man:
                    tokenizer.batch_encode(
                        batch_sequences=[''],
                        max_seq_len=invalid_input
                    )

                if isinstance(cxt_man.exception, TypeError):
                    self.assertEqual(
                        cxt_man.exception.args[0],
                        '`max_seq_len` must be an instance of `int`.',
                        msg=msg2
                    )
                else:
                    self.assertEqual(
                        cxt_man.exception.args[0],
                        '`max_seq_len` must be greater than `1` or equal to '
                        '`-1`.',
                        msg=msg2
                    )

    def test_return_type(self):
        r"""Return `List[List[int]]`."""
        msg = 'Must return `List[List[int]]`.'
        examples = (
            ['Hello World!', 'I am a legend.', 'y = f(x)'],
            ['Hello World!', '', ''],
            ['', 'I am a legend.', ''],
            ['', '', 'y = f(x)'],
            ['', '', ''],
            [],
        )

        for batch_sequences in examples:
            for tokenizer in self.tokenizers:
                batch_token_ids = tokenizer.batch_encode(
                    batch_sequences=batch_sequences
                )
                self.assertIsInstance(batch_token_ids, list, msg=msg)
                for token_ids in batch_token_ids:
                    self.assertIsInstance(token_ids, list, msg=msg)
                    for token_id in token_ids:
                        self.assertIsInstance(token_id, int, msg=msg)

    def test_encode_format(self):
        r"""Follow encode format."""
        msg = (
            'Must follow encode format: '
            '[bos] t1 t2 ... tn [eos] [pad] ... [pad].'
        )
        examples = (
            (
                ['Hello World!', 'I am a legend.', 'y = f(x)'],
                [
                    [0, 10, 6, 4, 4, 7, 5, 11, 7, 12, 4, 8, 13, 1, 2, 2],
                    [0, 14, 5, 9, 15, 5, 9, 5, 4, 6, 16, 6, 17, 8, 18, 1],
                    [0, 3, 5, 3, 5, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2],
                ],
            ),
            (
                ['Hello World!', '', ''],
                [
                    [0, 10, 6, 4, 4, 7, 5, 11, 7, 12, 4, 8, 13, 1],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
            ),
            (
                ['', 'I am a legend.', ''],
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 14, 5, 9, 15, 5, 9, 5, 4, 6, 16, 6, 17, 8, 18, 1],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
            ),
            (
                ['', '', 'y = f(x)'],
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 3, 5, 3, 5, 3, 3, 3, 3, 1],
                ],
            ),
            (
                ['', '', ''],
                [
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ],
            ),
            ([], [],),
        )

        for batch_sequences, batch_token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_encode(batch_sequences=batch_sequences),
                    batch_token_ids,
                    msg=msg
                )

    def test_truncate(self):
        r"""Batch token ids' length must not exceed `max_seq_len`."""
        msg = 'Batch token ids\' length must not exceed `max_seq_len`.'
        examples = (
            (
                ['Hello World!', 'I am a legend.', 'y = f(x)'],
                [
                    [0, 10, 6, 4, 4, 7, 5, 11, 7, 1],
                    [0, 14, 5, 9, 15, 5, 9, 5, 4, 1],
                    [0, 3, 5, 3, 5, 3, 3, 3, 3, 1],
                ],
                10,
            ),
            (
                ['Hello World!', 'I am a legend.', 'y = f(x)'],
                [
                    [0, 10, 6, 4, 1],
                    [0, 14, 5, 9, 1],
                    [0, 3, 5, 3, 1],
                ],
                5,
            ),
            (
                ['Hello World!', 'I am a legend.', 'y = f(x)'],
                [
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ],
                2,
            ),
            (
                ['', '', ''],
                [
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ],
                2,
            ),
            ([], [], 2),
        )

        for batch_sequences, batch_token_ids, max_seq_len in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_encode(
                        batch_sequences=batch_sequences,
                        max_seq_len=max_seq_len
                    ),
                    batch_token_ids,
                    msg=msg
                )

    def test_padding(self):
        r"""Batch token ids' length must pad to `max_seq_len`."""
        msg = 'Batch token ids\' length must pad to `max_seq_len`.'
        examples = (
            (
                ['Hello World!', 'I am a legend.', 'y = f(x)'],
                [
                    [
                        0, 10, 6, 4, 4, 7, 5, 11, 7, 12,
                        4, 8, 13, 1, 2, 2, 2, 2, 2, 2
                    ],
                    [
                        0, 14, 5, 9, 15, 5, 9, 5, 4, 6,
                        16, 6, 17, 8, 18, 1, 2, 2, 2, 2
                    ],
                    [
                        0, 3, 5, 3, 5, 3, 3, 3, 3, 1,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2
                    ],
                ],
                20,
            ),
            (
                ['Hello World!', '', ''],
                [
                    [0, 10, 6, 4, 4, 7, 5, 11, 7, 12, 4, 8, 13, 1, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
                15,
            ),
            (
                ['', 'I am a legend.', ''],
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 14, 5, 9, 15, 5, 9, 5, 4, 6, 16, 6, 17, 8, 18, 1, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
                17,
            ),
            (
                ['', '', 'y = f(x)'],
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 3, 5, 3, 5, 3, 3, 3, 3, 1, 2, 2],
                ],
                12,
            ),
            (
                ['', '', ''],
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
                10,
            ),
            ([], [], 100),
        )

        for batch_sequences, batch_token_ids, max_seq_len in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_encode(
                        batch_sequences=batch_sequences,
                        max_seq_len=max_seq_len
                    ),
                    batch_token_ids,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
