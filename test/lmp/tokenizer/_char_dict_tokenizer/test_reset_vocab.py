r"""Test `lmp.tokenizer.CharDictTokenizer.reset_vocab`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_dict_tokenizer/test_reset_vocab.py
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
    r"""Test case for `lmp.tokenizer.CharDictTokenizer.reset_vocab`."""

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

    def test_reset_token_to_id_and_id_to_token(self):
        r"""Reset `token_to_id` and `id_to_token`."""
        msg = 'Must reset `token_to_id` and `id_to_token`.'

        token_to_id = {
            '[bos]': 0,
            '[eos]': 1,
            '[pad]': 2,
            '[unk]': 3
        }
        id_to_token = {
            0: '[bos]',
            1: '[eos]',
            2: '[pad]',
            3: '[unk]'
        }

        for tokenizer in self.tokenizers:
            tokenizer.build_vocab(['Hello World!', 'I am a legend.'])
            tokenizer.reset_vocab()
            self.assertTrue(hasattr(tokenizer, 'token_to_id'))
            self.assertIsInstance(
                tokenizer.token_to_id,
                type(token_to_id),
                msg=msg
            )
            self.assertTrue(hasattr(tokenizer, 'id_to_token'))
            self.assertIsInstance(
                tokenizer.id_to_token,
                type(id_to_token),
                msg=msg
            )
            self.assertEqual(tokenizer.token_to_id, token_to_id, msg=msg)
            self.assertEqual(tokenizer.id_to_token, id_to_token, msg=msg)


if __name__ == '__main__':
    unittest.main()
