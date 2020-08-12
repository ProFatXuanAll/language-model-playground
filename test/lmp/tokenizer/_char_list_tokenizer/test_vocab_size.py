r"""Test `lmp.tokenizer.CharListTokenizer.vocab_size`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_char_list_tokenizer/test_vocab_size.py
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

from lmp.tokenizer import CharListTokenizer


class TestVocabSize(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.CharListTokenizer.vocab_size`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharListTokenizer()
        self.uncased_tokenizer = CharListTokenizer(is_uncased=True)
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
            inspect.isdatadescriptor(CharListTokenizer.vocab_size),
            msg=msg
        )
        self.assertFalse(
            inspect.isfunction(CharListTokenizer.vocab_size),
            msg=msg
        )
        self.assertFalse(
            inspect.ismethod(CharListTokenizer.vocab_size),
            msg=msg
        )

    def test_return_type(self):
        r"""Return `int`"""
        msg = 'Must return `int`.'

        for tokenizer in self.tokenizers:
            self.assertIsInstance(tokenizer.vocab_size, int, msg=msg)

    def test_return_value(self):
        r"""Return vocabulary size."""
        msg = 'Inconsistent vocabulary size.'

        for tokenizer in self.tokenizers:
            self.assertEqual(tokenizer.vocab_size, 4, msg=msg)

    def test_increase_vocab_size(self):
        r"""Increase vocabulary size after `build_vocab`."""
        msg = 'Must increase vocabulary size after `build_vocab`.'
        examples = (
            (('HeLlO WoRlD!', 'I aM a LeGeNd.'), 18, 15),
            (('y = f(x)',), 24, 21),
            (('',), 24, 21),
        )

        sp_tokens_size = len(list(CharListTokenizer.special_tokens()))

        for batch_sequences, cased_vocab_size, uncased_vocab_size in examples:
            self.cased_tokenizer.build_vocab(batch_sequences)
            self.assertEqual(
                self.cased_tokenizer.vocab_size,
                cased_vocab_size + sp_tokens_size,
                msg=msg
            )
            self.uncased_tokenizer.build_vocab(batch_sequences)
            self.assertEqual(
                self.uncased_tokenizer.vocab_size,
                uncased_vocab_size + sp_tokens_size,
                msg=msg
            )

    def test_reset_vocab_size(self):
        r"""Reset vocabulary size after `reset_vocab`."""
        msg = 'Must reset vocabulary size after `reset_vocab`.'
        examples = (
            ('HeLlO WoRlD!', 'I aM a LeGeNd.'),
            ('y = f(x)',),
            ('',),
        )

        sp_tokens_size = len(list(CharListTokenizer.special_tokens()))

        for batch_sequences in examples:
            for tokenizer in self.tokenizers:
                tokenizer.build_vocab(batch_sequences)
                tokenizer.reset_vocab()
                self.assertEqual(
                    tokenizer.vocab_size,
                    sp_tokens_size,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
