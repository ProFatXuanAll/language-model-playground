r"""Test `lmp.tokenizer.BaseListTokenizer.encode`.

Usage:
    python -m unittest test.lmp.tokenizer._base_list_tokenizer.test_encode
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

from typing import List

# self-made modules

from lmp.tokenizer import BaseListTokenizer


class TestEncode(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseListTokenizer.encode`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseListTokenizer.encode),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        name='sequence',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=-1
                    )
                ],
                return_annotation=List[int]
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

        for is_uncased in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                BaseListTokenizer(is_uncased=is_uncased).encode('H')

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `BaseListTokenizer`: '
                'method `tokenize` not implemented yet.',
                msg=msg2
            )


if __name__ == '__main__':
    unittest.main()
