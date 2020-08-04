r"""Test `lmp.tokenizer.CharDictTokenizer.convert_token_to_id`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/test_convert_token_to_id.py
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


class TestConvertTokenToId(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.convert_token_to_id`."""

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
            inspect.signature(CharDictTokenizer.convert_token_to_id),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='token',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=int
            ),
            msg=msg
        )

    def test_invalid_input_token(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False, b'',
            [], (), {}, set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.convert_token_to_id(token=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`token` must be instance of `str`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `int`."""
        msg = 'Must return `int`.'
        examples = (
            'H',
            '',
        )

        for token in examples:
            for tokenizer in self.tokenizers:
                token_id = tokenizer.convert_token_to_id(token=token)
                self.assertIsInstance(token_id, int, msg=msg)

    def test_convert_unknown_token_to_id(self):
        r"""Return `int` must be [UNK] id."""
        msg = 'Must return [UNK] id.'
        examples = (
            (
                'H',
                3
            ),
        )

        for token, ans_token_id in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_token_to_id(token=token),
                    ans_token_id,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
