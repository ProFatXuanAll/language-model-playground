r"""Test `lmp.tokenizer.CharListTokenizer.__init__`.

Usage:
    python -m unittest test.lmp.tokenizer._char_list_tokenizer.test_init
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
from lmp.tokenizer import CharListTokenizer


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharListTokenizer.__init__`."""

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
            inspect.signature(CharListTokenizer.__init__),
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

    def test_inheritance(self):
        r""""Is subclass of `lmp.tokenizer.BaseListTokenizer`."""
        msg = 'Must be subclass of `lmp.tokenizer.BaseListTokenizer`.'

        for tokenizer in self.tokenizers:
            self.assertIsInstance(tokenizer, BaseListTokenizer, msg=msg)

    def test_invalid_input_is_uncased(self):
        r"""Raise `TypeError` when input `is_uncased` is invalid."""
        msg1 = 'Must raise `TypeError` when input `is_uncased` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j,
            1j, '', b'', (), [], {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                CharListTokenizer(is_uncased=invalid_input)

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
            ('bos_token', '[bos]'),
            ('eos_token', '[eos]'),
            ('pad_token', '[pad]'),
            ('unk_token', '[unk]'),
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

    def test_instance_attribute(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'

        examples = (
            ('is_uncased', bool),
            ('token_to_id', list),
        )

        for attr, attr_type in examples:
            for tokenizer in self.tokenizers:
                self.assertTrue(
                    hasattr(tokenizer, attr),
                    msg=msg1.format(attr)
                )

                self.assertIsInstance(
                    getattr(tokenizer, attr),
                    attr_type,
                    msg=msg2.format(attr, attr_type.__name__)
                )


if __name__ == '__main__':
    unittest.main()
