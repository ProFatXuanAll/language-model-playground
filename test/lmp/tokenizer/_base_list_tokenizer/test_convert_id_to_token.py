r"""Test `lmp.tokenizer.BaseListTokenizer.convert_id_to_token`.

Usage:
    python -m unittest \
        test.lmp.tokenizer._base_list_tokenizer.test_convert_id_to_token
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

# self-made modules

from lmp.tokenizer import BaseListTokenizer


class TestConvertIdToToken(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseListTokenizer.convert_id_to_token`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = BaseListTokenizer()
        self.uncased_tokenizer = BaseListTokenizer(is_uncased=True)
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
            inspect.signature(BaseListTokenizer.convert_id_to_token),
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
        r"""Raise `TypeError` when input `token_id` is invalid."""
        msg1 = 'Must raise `TypeError` when input `token_id` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.convert_id_to_token(token_id=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`token_id` must be an instance of `int`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `str`."""
        msg = 'Must return `str`.'
        examples = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        for token_id in examples:
            for tokenizer in self.tokenizers:
                self.assertIsInstance(
                    tokenizer.convert_id_to_token(token_id=token_id),
                    str,
                    msg=msg
                )

    def test_return_special_token(self):
        r"""Return special token."""
        msg = 'Must return special token.'
        examples = (
            (0, '[bos]'),
            (1, '[eos]'),
            (2, '[pad]'),
            (3, '[unk]'),
        )

        for token_id, ans_token in examples:
            for tokenizer in self.tokenizers:
                self.assertEqual(
                    tokenizer.convert_id_to_token(token_id=token_id),
                    ans_token,
                    msg=msg
                )

    def test_return_unknown_token(self):
        r"""Return unknown token when token id is unknown."""
        msg = 'Must return unknown token when token id is unknown.'
        examples = (
            (4, '[unk]'),
            (5, '[unk]'),
            (6, '[unk]'),
            (7, '[unk]'),
            (8, '[unk]'),
            (9, '[unk]'),
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
