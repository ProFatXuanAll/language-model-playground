r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.convert_tokens_to_ids`.

Usage:
    python -m unittest \
        test.lmp.tokenizer._whitespace_dict_tokenizer.test_convert_tokens_to_ids
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

from lmp.tokenizer import WhitespaceDictTokenizer


class TestConvertTokensToIds(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceDictTokenizer.convert_tokens_to_ids`."""

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
            inspect.signature(WhitespaceDictTokenizer.convert_tokens_to_ids),
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
                    ),
                ],
                return_annotation=List[int]
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
            [b''], [object()], [lambda x: x], [type], [None], [NotImplemented],
            [...], ['', False], ['', True], ['', 0], ['', 1], ['', -1],
            ['', 0.0], ['', 1.0], ['', math.nan], ['', -math.nan],
            ['', math.inf], ['', -math.inf], ['', 0j], ['', 1j], ['', b''],
            ['', object()], ['', lambda x: x], ['', type], ['', None],
            ['', NotImplemented], ['', ...],
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.convert_tokens_to_ids(tokens=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`tokens` must be an instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[int]`."""
        msg = 'Must return `List[int]`.'
        examples = (
            ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd'],
            ['H'],
            [''],
            [],
        )

        for tokens in examples:
            for tokenizer in self.tokenizers:
                token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
                self.assertIsInstance(token_ids, list, msg=msg)
                for token_id in token_ids:
                    self.assertIsInstance(token_id, int, msg=msg)

    def test_return_special_token_ids(self):
        r"""Return special token ids."""
        msg = 'Must return special token ids.'
        examples = (
            (
                ['[bos]', '[eos]', '[pad]', '[unk]'],
                [0, 1, 2, 3],
            ),
            (
                ['[bos]'],
                [0],
            ),
            (
                ['[eos]'],
                [1],
            ),
            (
                ['[pad]'],
                [2],
            ),
            (
                ['[unk]'],
                [3],
            ),
        )

        for tokens, ans_token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_tokens_to_ids(tokens=tokens),
                    ans_token_ids,
                    msg=msg
                )

    def test_return_unknown_token_ids(self):
        r"""Return unknown token ids when tokens are unknown."""
        msg = 'Must return unknown token ids when tokens are unknown.'
        examples = (
            (
                ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd'],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ),
            (
                [
                    '[bos]', 'H', 'e', 'l', 'l', 'o', ' ',
                    'W', 'o', 'r', 'l', 'd', '[eos]', '[pad]', '[pad]',
                ],
                [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2],
            ),
        )

        for tokens, ans_token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_tokens_to_ids(tokens=tokens),
                    ans_token_ids,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
