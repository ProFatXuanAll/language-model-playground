r"""Test `lmp.tokenizer.BaseListTokenizer.reset_vocab`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_list_tokenizer/test_reset_vocab.py
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

from lmp.tokenizer import BaseListTokenizer


class TestResetVocab(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.BaseListTokenizer.reset_vocab`."""

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
            inspect.signature(BaseListTokenizer.reset_vocab),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_token_to_id_existence(self):
        r"Test whether create `token_to_id` in `reset_vocab`"
        msg1 = '`token_to_id` must be `list`'
        msg2 = 'Inconsistent error message.'

        token_to_id = ['[BOS]', '[EOS]', '[PAD]', '[UNK]']

        for tokenizer in self.tokenizers:
            self.assertIsInstance(tokenizer.token_to_id, list, msg=msg1)
            self.assertEqual(tokenizer.token_to_id, token_to_id, msg=msg2)


if __name__ == '__main__':
    unittest.main()
