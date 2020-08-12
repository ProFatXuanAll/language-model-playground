r"""Test `lmp.tokenizer.BaseTokenizer.normalize`.

Usage:
    python -m unittest test/lmp/tokenizer/_base_tokenizer/test_normalize.py
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


class TestNormalize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.normalize`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseTokenizer.normalize),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='sequence',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str
                    )
                ],
                return_annotation=str
            ),
            msg=msg
        )


if __name__ == '__main__':
    unittest.main()
