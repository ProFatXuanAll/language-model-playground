r"""Test `lmp.tokenizer.CharDictTokenizer.convert_tokens_to_ids`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_dict_tokenizer/test_convert_tokens_to_ids.py
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


class TestConvertTokensToIds(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.convert_tokens_to_ids`."""

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
            inspect.signature(CharDictTokenizer.convert_tokens_to_ids),
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
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False,
            object(), lambda x: x, type, None, 0j, 1j, NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.convert_tokens_to_ids(tokens=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`tokens` must be instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `List[int]`."""
        msg = 'Must return `List[int]`.'
        examples = (
            ['H', 'e', 'l', 'l', 'o'],
            [''],
        )

        for tokens in examples:
            for tokenizer in self.tokenizers:
                token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
                self.assertIsInstance(token_ids, list, msg=msg)
                for token_id in token_ids:
                    self.assertIsInstance(token_id, int, msg=msg)

    def test_convert_unknown_tokens_to_ids(self):
        r"""Return `List[int]` must be [UNK] id."""
        msg = 'Must return [UNK] id.'
        examples = (
            (
                ['H', 'e', 'l', 'l', 'o'],
                [3, 3, 3, 3, 3]
            ),
        )

        for tokens, ans_token_ids in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_tokens_to_ids(tokens=tokens),
                    ans_token_ids,
                    msg=msg
                )

    def test_convert_special_tokens_to_ids(self):
        r"""Return `List[int]` must be [UNK] id."""
        msg = 'Must return [UNK] id.'
        examples = (
            (
                ['[BOS]', '[EOS]', '[PAD]', '[UNK]'],
                [0, 1, 2, 3]
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
