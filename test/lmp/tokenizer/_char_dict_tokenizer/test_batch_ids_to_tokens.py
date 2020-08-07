r"""Test `lmp.tokenizer.CharDictTokenizer.batch_ids_to_tokens`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_dict_tokenizer/test_batch_ids_to_tokens.py
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

from lmp.tokenizer import CharDictTokenizer


class TestBatchIdsToTokens(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharDictTokenizer.batch_ids_to_tokens`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharDictTokenizer()
        self.uncased_tokenizer = CharDictTokenizer(is_uncased=True)
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
            inspect.signature(CharDictTokenizer.batch_ids_to_tokens),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='batch_token_ids',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[Iterable[int]],
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=List[List[str]]
            ),
            msg=msg
        )

    def test_invalid_input_batch_token_ids(self):
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
            {None: 0}, {NotImplemented: 0}, {...: 0}, [['']], [[0.0]], [[1.0]],
            [[math.nan]], [[-math.nan]], [[math.inf]], [[-math.inf]], [[0j]],
            [[1j]], [[object()]], [[lambda x: x]], [[type]], [[None]],
            [[NotImplemented]], [[...]],
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.batch_ids_to_tokens(
                        batch_token_ids=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_token_ids` must be instance of `Iterable[Iterable[int]]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[List[str]]`."""
        msg = 'Must return `List[List[str]]`.'
        examples = (
            [
                [0, 1, 2, 3, 4],
                [3, 3, 3, 3, 3]
            ],
        )

        for batch_token_ids in examples:
            for tokenizer in self.tokenizers:
                batch_tokens = tokenizer.batch_ids_to_tokens(
                    batch_token_ids=batch_token_ids
                )
                self.assertIsInstance(batch_tokens, list, msg=msg)
                for tokens in batch_tokens:
                    self.assertIsInstance(tokens, list, msg=msg)
                    for token in tokens:
                        self.assertIsInstance(token, str, msg=msg)

    def test_function(self):
        r"""Test token id inverse look up on batch of tokens' ids."""
        msg = 'Inconsistent error message.'

        examples = (
            (
                [
                    [0, 1, 2, 3, 4],
                    [3, 3, 3, 3, 3]
                ],
                [
                    ['[BOS]', '[EOS]', '[PAD]', '[UNK]', '[UNK]'],
                    ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]']
                ]
            ),
            (
                [[]],
                [[]]
            ),
        )

        for batch_token_ids, ans_batch_tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_ids_to_tokens(
                        batch_token_ids=batch_token_ids
                    ),
                    ans_batch_tokens,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
