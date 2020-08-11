r"""Test `lmp.tokenizer.CharDictTokenizer.__init__`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_dict_tokenizer/test_init.py
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

from lmp.tokenizer import CharDictTokenizer


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharDictTokenizer.__init__`."""

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
            inspect.signature(CharDictTokenizer.__init__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='is_uncased',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=False
                    )
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, '', b'', 0j, 1j,
            [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                CharDictTokenizer(is_uncased=invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`is_uncased` must be an instance of `bool`.',
                msg=msg2
            )

    def test_class_attributes(self):
        r"""Declare required class attributes."""
        msg1 = 'Missing class attribute `{}`.'
        msg2 = 'Class attribute `{}` must be an instance of `{}`.'
        msg3 = 'Class attribute `{}` must be `{}`.'

        examples = (
            ('bos_token', '[BOS]'),
            ('eos_token', '[EOS]'),
            ('pad_token', '[PAD]'),
            ('unk_token', '[UNK]'),
        )

        for attr, attr_val in examples:
            for tokenizer in self.tokenizers:
                self.assertTrue(
                    hasattr(tokenizer, attr),
                    msg=msg1.format(attr)
                )

                self.assertIsInstance(
                    getattr(tokenizer, attr),
                    type(attr_val),
                    msg=msg2.format(attr, type(attr_val).__name__)
                )

                self.assertEqual(
                    getattr(tokenizer, attr),
                    attr_val,
                    msg=msg3.format(attr, attr_val)
                )


if __name__ == '__main__':
    unittest.main()
