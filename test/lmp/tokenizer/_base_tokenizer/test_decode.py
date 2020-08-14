r"""Test `lmp.tokenizer.BaseTokenizer.decode`.

Usage:
    python -m unittest test/lmp/tokenizer/_base_tokenizer/test_decode.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

from typing import Iterable

# self-made modules

from lmp.tokenizer import BaseTokenizer


class TestDecode(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.decode`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseTokenizer.decode),
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
                    inspect.Parameter(
                        name='remove_special_tokens',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=False
                    ),
                ],
                return_annotation=str
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
            r"""Intented to not implement `convert_id_to_token`."""

            def reset_vocab(self):
                pass
        # pylint: enable=W0231
        # pylint: enable=W0223

        for is_uncased in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                SubClassTokenizer(is_uncased=is_uncased).decode([0])

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `SubClassTokenizer`: '
                'method `convert_id_to_token` not implemented yet.',
                msg=msg2
            )


if __name__ == '__main__':
    unittest.main()
