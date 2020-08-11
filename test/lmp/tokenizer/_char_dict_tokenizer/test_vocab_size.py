r"""Test `lmp.tokenizer.CharDictTokenizer.vocab_size`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_dict_tokenizer/test_vocab_size.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import unittest

# self-made modules

from lmp.tokenizer import CharDictTokenizer


class TestVocabSize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharDictTokenizer.vocab_size`."""

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
        msg = 'Inconsistent property signature.'

        self.assertTrue(
            inspect.isdatadescriptor(CharDictTokenizer.vocab_size),
            msg=msg
        )
        self.assertFalse(
            inspect.isfunction(CharDictTokenizer.vocab_size),
            msg=msg
        )
        self.assertFalse(
            inspect.ismethod(CharDictTokenizer.vocab_size),
            msg=msg
        )

    def test_expected_return(self):
        r"""Return expected number."""
        msg = 'Inconsistent vocabulary size.'

        for tokenizer in self.tokenizers:

            self.assertEqual(
                tokenizer.vocab_size,
                len(tokenizer.token_to_id),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
