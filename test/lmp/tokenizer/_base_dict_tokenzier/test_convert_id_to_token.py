r"""Test `lmp.tokenizer.CharDictTokenizer.convert_id_to_token`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/convert_id_to_token.py
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

# self-made modules

from lmp.tokenizer import CharDictTokenizer


class TestConvertIdToToken(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.convert_id_to_token`."""

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
            inspect.signature(CharDictTokenizer.convert_id_to_token),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='token_id',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=str
            ),
            msg=msg
        )

    def test_invalid_input_token_id(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, math.inf, b'',
            [], (), {}, set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.convert_id_to_token(token_id=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`token_id` must be instance of `int`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
        examples = (
            0, 1, 2,
        )

        for token_id in examples:
            for tokenizer in self.tokenizers:
                token = tokenizer.convert_id_to_token(token_id=token_id)
                self.assertIsInstance(token, str, msg=msg)

    def test_convert_special_and_unknown_id_to_token(self):
        r"""Return `str`."""
        msg = 'Must return token str.'
        examples = (
            (
                0,
                '[BOS]'
            ),
            (
                1,
                '[EOS]'
            ),
            (
                2,
                '[PAD]'
            ),
            (
                3,
                '[UNK]'
            ),
            (
                6,
                '[UNK]'
            ),
        )

        for token_id, ans_token in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_id_to_token(token_id=token_id),
                    ans_token,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
