r"""Test `lmp.tokenizer.CharListTokenizer.convert_ids_to_tokens`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_list_tokenizer/test_convert_ids_to_tokens.py
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


class TestConvertIdsToTokens(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharListTokenizer.convert_ids_to_tokens`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharListTokenizer()
        self.uncased_tokenizer = CharListTokenizer(is_uncased=True)
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
            inspect.signature(CharListTokenizer.convert_ids_to_tokens),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='token_ids',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[int],
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=List[str]
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
            [math.inf], [-math.inf], [0j], [1j], [''], [b''], [object()],
            [lambda x: x], [type], [None], [NotImplemented], [...], [0, 0.0],
            [0, 1.0], [0, math.nan], [0, -math.nan], [0, math.inf],
            [0, -math.inf], [0, 0j], [0, 1j], [0, ''], [0, b''], [0, object()],
            [0, lambda x: x], [0, type], [0, None], [0, NotImplemented],
            [0, ...],
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.convert_ids_to_tokens(token_ids=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`token_ids` must be an instance of `Iterable[int]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        examples = (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0],
            [],
        )

        for token_ids in examples:
            for tokenizer in self.tokenizers:
                tokens = tokenizer.convert_ids_to_tokens(token_ids=token_ids)
                self.assertIsInstance(tokens, list, msg=msg)
                for token in tokens:
                    self.assertIsInstance(token, str, msg=msg)

    def test_return_special_tokens(self):
        r"""Return special tokens."""
        msg = 'Must return special tokens.'
        examples = (
            (
                [0, 1, 2, 3],
                ['[bos]', '[eos]', '[pad]', '[unk]'],
            ),
            (
                [0],
                ['[bos]'],
            ),
            (
                [1],
                ['[eos]'],
            ),
            (
                [2],
                ['[pad]'],
            ),
            (
                [3],
                ['[unk]'],
            ),
        )

        for token_ids, ans_tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_ids_to_tokens(token_ids=token_ids),
                    ans_tokens,
                    msg=msg
                )

    def test_return_unknown_tokens(self):
        r"""Return unknown tokens when token ids are unknown."""
        msg = 'Must return unknown tokens when token ids are unknown.'
        examples = (
            (
                [4, 5, 6, 7, 8, 9],
                ['[unk]', '[unk]', '[unk]', '[unk]', '[unk]', '[unk]'],
            ),
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [
                    '[bos]', '[eos]', '[pad]', '[unk]', '[unk]',
                    '[unk]', '[unk]', '[unk]', '[unk]', '[unk]',
                ],
            ),
        )

        for token_ids, ans_tokens in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_ids_to_tokens(token_ids=token_ids),
                    ans_tokens,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
