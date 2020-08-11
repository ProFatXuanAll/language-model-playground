r"""Test `lmp.tokenizer.BaseTokenizer.convert_token_to_id`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_tokenizer/test_convert_token_to_id.py
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


class TestConvertTokenToId(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.convert_token_to_id`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseTokenizer.convert_token_to_id),
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

    def test_abstract_method(self):
        r"""Raise `NotImplementedError` when subclass did not implement."""
        msg1 = (
            'Must raise `NotImplementedError` when subclass did not implement.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (True, False)

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseTokenizer):
            r"""Intented to not implement `convert_token_to_id`."""

            def reset_vocab(self):
                pass
        # pylint: enable=W0231
        # pylint: enable=W0223

        for is_uncased in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                SubClassTokenizer(
                    is_uncased=is_uncased
                ).convert_token_to_id('')

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `SubClassTokenizer`: '
                'method `convert_token_to_id` not implemented yet.',
                msg=msg2
            )


if __name__ == '__main__':
    unittest.main()
