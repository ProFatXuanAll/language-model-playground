r"""Test `lmp.tokenizer.BaseDictTokenizer.batch_tokens_to_ids`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/test_batch_tokens_to_ids.py
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

from lmp.tokenizer import BaseDictTokenizer


class TestBatchTokensToIds(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseDictTokenizer.batch_tokens_to_ids`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = BaseDictTokenizer()
        self.uncased_tokenizer = BaseDictTokenizer(is_uncased=True)
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
            inspect.signature(BaseDictTokenizer.batch_tokens_to_ids),
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
                return_annotation=List[List[int]]
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
                    tokenizer.batch_tokens_to_ids(batch_tokens=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_tokens` must be instance of `Iterable[Iterable[str]]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[List[str]]`."""
        msg = 'Must return `List[List[int]]`.'
        examples = (
            [
                ['H', 'e', 'l', 'l', 'o'],
                ['[BOS]', '[EOS]', '[PAD]', '[UNK]']
            ],
        )

        for batch_tokens in examples:
            for tokenizer in self.tokenizers:
                batch_token_ids = tokenizer.batch_tokens_to_ids(
                    batch_tokens=batch_tokens
                )
                self.assertIsInstance(batch_token_ids, list, msg=msg)
                for token_ids in batch_token_ids:
                    self.assertIsInstance(token_ids, list, msg=msg)
                    for token_id in token_ids:
                        self.assertIsInstance(token_id, int, msg=msg)

    def test_function(self):
        r"""Test token id look up on batch of tokens."""
        msg = 'Inconsistent error message.'

        examples = (
            (
                [
                    ['H', 'e', 'l', 'l', 'o'],
                    ['[BOS]', '[EOS]', '[PAD]', '[UNK]']
                ],
                [
                    [3, 3, 3, 3, 3],
                    [0, 1, 2, 3]
                ]
            ),
            (
                [['']],
                [[3]]
            ),
        )

        for batch_tokens, ans_batch_token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_tokens_to_ids(
                        batch_tokens=batch_tokens
                    ),
                    ans_batch_token_ids,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
