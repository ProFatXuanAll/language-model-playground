r"""Test `lmp.tokenizer.BaseTokenizer.__init__`.

Usage:
    python -m unittest test/lmp/tokenizer/_base_tokenizer/test_init.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

# self-made modules

from lmp.tokenizer import BaseTokenizer


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.__init__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseTokenizer.__init__),
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

    def test_abstract_class(self):
        r"""Raise `NotImplementedError` when construct instance."""
        msg1 = 'Must raise `NotImplementedError` when construct instance.'
        msg2 = 'Inconsistent error message.'
        examples = (True, False)

        for is_uncased in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                BaseTokenizer(is_uncased=is_uncased)

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `BaseTokenizer`: '
                'method `reset_vocab` not implemented yet.',
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
            self.assertTrue(
                hasattr(BaseTokenizer, attr),
                msg=msg1.format(attr)
            )

            self.assertIsInstance(
                getattr(BaseTokenizer, attr),
                type(attr_val),
                msg=msg2.format(attr, type(attr_val).__name__)
            )

            self.assertEqual(
                getattr(BaseTokenizer, attr),
                attr_val,
                msg=msg3.format(attr, attr_val)
            )


if __name__ == '__main__':
    unittest.main()
