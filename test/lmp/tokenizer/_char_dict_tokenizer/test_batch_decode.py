r"""Test `lmp.tokenizer.CharDictTokenizer.batch_decode`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_dict_tokenizer/test_batch_decode.py
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

from typing import Iterable
from typing import List

# self-made modules

from lmp.tokenizer import CharDictTokenizer


class TestBatchDecode(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.batch_decode`."""

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
            inspect.signature(CharDictTokenizer.batch_decode),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        name='batch_token_ids',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[Iterable[int]],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='remove_special_tokens',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=False
                    )
                ],
                return_annotation=List[str]
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
                    tokenizer.batch_decode(batch_token_ids=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_token_ids` must be instance of `Iterable[Iterable[int]]`.',
                    msg=msg2
                )

    def test_invalid_input_remove_special_tokens(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, b'', 0j, 1j,
            [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.batch_decode(
                        batch_token_ids=[[0, 1, 2, 3, 4]],
                        remove_special_tokens=invalid_input
                    )

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`remove_special_tokens` must be instance of `bool`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        examples = (
            [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8]
            ],
            [],
        )

        for batch_token_ids in examples:
            for tokenizer in self.tokenizers:
                batch_sequences = tokenizer.batch_decode(
                    batch_token_ids=batch_token_ids
                )
                self.assertIsInstance(batch_sequences, list, msg=msg)
                for sequence in batch_sequences:
                    self.assertIsInstance(sequence, str, msg=msg)

    def test_remove_special_tokens(self):
        r"""Return List[str] must remove special token."""
        msg = (
            'Return result must remove special token.'
        )
        examples = (
            (
                [
                    [0, 3, 3, 3, 3, 3, 1],
                    [0, 3, 3, 1]
                ],
                [
                    '[UNK][UNK][UNK][UNK][UNK]',
                    '[UNK][UNK]'
                ]
            ),
        )

        for batch_token_ids, ans_batch_sequence in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.batch_decode(
                        batch_token_ids=batch_token_ids,
                        remove_special_tokens=True
                    ),
                    ans_batch_sequence,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
