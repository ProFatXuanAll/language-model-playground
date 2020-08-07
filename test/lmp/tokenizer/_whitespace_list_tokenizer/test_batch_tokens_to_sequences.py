r"""Test `lmp.tokenizer.WhitespaceListTokenizer.batch_tokens_to_sequences`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_list_tokenizer/test_batch_tokens_to_sequences.py
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

from lmp.tokenizer import WhitespaceListTokenizer


class TestBatchTokensToSequences(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceListTokenizer.batch_tokens_to_sequences`."""

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
            inspect.signature(
                WhitespaceListTokenizer.batch_tokens_to_sequences),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='batch_tokens',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[Iterable[str]],
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=List[str]
            ),
            msg=msg
        )

    def test_invalid_input_batch_tokens(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ..., (False,), (True,), (0,), (1,), (-1,), (0.0,),
            (1.0,), (math.nan,), (-math.nan,), (math.inf,), (-math.inf,),
            (0j,), (1j,), (object(),), (lambda x: x,), (type,), (None,),
            (NotImplemented,), (...,), [False], [True], [0], [1], [-1], [0.0],
            [1.0], [math.nan], [-math.nan], [math.inf], [-math.inf], [0j],
            [1j], [object()], [lambda x: x], [type], [None], [NotImplemented],
            [...], {False}, {True}, {0}, {1}, {-1}, {0.0}, {1.0}, {math.nan},
            {-math.nan}, {math.inf}, {-math.inf}, {0j}, {1j}, {object()},
            {lambda x: x}, {type}, {None}, {NotImplemented}, {...},
            {False: 0}, {True: 0}, {0: 0}, {1: 0}, {-1: 0}, {0.0: 0}, {1.0: 0},
            {math.nan: 0}, {-math.nan: 0}, {math.inf: 0}, {-math.inf: 0},
            {0j: 0}, {1j: 0}, {object(): 0}, {lambda x: x: 0}, {type: 0},
            {None: 0}, {NotImplemented: 0}, {...: 0}, [[0.0]], [[1.0]],
            [[math.nan]], [[-math.nan]], [[math.inf]], [[-math.inf]], [[0j]],
            [[1j]], [[object()]], [[lambda x: x]], [[type]], [[None]],
            [[NotImplemented]], [[...]],
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.batch_tokens_to_sequences(
                        batch_tokens=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_tokens` must be instance of `Iterable[Iterable[str]]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        examples = (
            [
                ['H', 'e', 'l', 'l', 'o'],
                ['[BOS]', '[EOS]', '[PAD]', '[UNK]']
            ],
        )

        for batch_tokens in examples:
            for tokenizer in self.tokenizers:
                batch_sequences = tokenizer.batch_tokens_to_sequences(
                    batch_tokens=batch_tokens
                )
                self.assertIsInstance(batch_sequences, list, msg=msg)
                for sequence in batch_sequences:
                    self.assertIsInstance(sequence, str, msg=msg)

    def test_function(self):
        r"""Test batch of tokens back to sequences."""
        msg = 'Inconsistent error message.'

        examples = (
            (
                [
                    ['H', 'e', 'l', 'l', 'o'],
                    ['[BOS]', '[EOS]', '[PAD]', '[UNK]']
                ],
                [
                    'H e l l o',
                    '[BOS] [EOS] [PAD] [UNK]'
                ]
            ),
            (
                [['']],
                ['']
            ),
        )

        for batch_tokens, ans_batch_seqs in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_tokens_to_sequences(
                        batch_tokens=batch_tokens
                    ),
                    ans_batch_seqs,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
