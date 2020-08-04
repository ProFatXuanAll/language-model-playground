r"""Test `lmp.tokenizer.CharDictTokenizer.reset_vocab`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/test_reset_vocab.py
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


class TestResetVocab(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.reset_vocab`."""

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
            inspect.signature(CharDictTokenizer.reset_vocab),
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

        token_to_id = {'[BOS]': 0, '[EOS]': 1, '[PAD]': 2, '[UNK]': 3}

        for tokenizer in self.tokenizers:
            self.assertIsInstance(tokenizer.token_to_id, dict, msg=msg1)
            self.assertEqual(tokenizer.token_to_id, token_to_id, msg=msg2)

    def test_id_to_token_existence(self):
        r"Test whether create `id_to_token` in `reset_vocab`"
        msg1 = '`id_to_token` must be `dict`'
        msg2 = 'Inconsistent error message.'

        id_to_token = {0: '[BOS]', 1: '[EOS]', 2: '[PAD]', 3: '[UNK]'}

        for tokenizer in self.tokenizers:
            self.assertIsInstance(tokenizer.id_to_token, dict, msg=msg1)
            self.assertEqual(tokenizer.id_to_token, id_to_token, msg=msg2)


if __name__ == '__main__':
    unittest.main()
