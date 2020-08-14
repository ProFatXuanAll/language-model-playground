r"""Test `lmp.tokenizer.BaseTokenizer.vocab_size`.

Usage:
    python -m unittest test/lmp/tokenizer/_base_tokenizer/test_vocab_size.py
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


class TestVocabSize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.vocab_size`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent property signature.'

        self.assertTrue(
            inspect.isdatadescriptor(BaseTokenizer.vocab_size),
            msg=msg
        )
        self.assertFalse(
            inspect.isfunction(BaseTokenizer.vocab_size),
            msg=msg
        )
        self.assertFalse(
            inspect.ismethod(BaseTokenizer.vocab_size),
            msg=msg
        )


if __name__ == '__main__':
    unittest.main()
