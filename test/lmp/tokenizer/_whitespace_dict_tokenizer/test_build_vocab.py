r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.build_vocab`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_dict_tokenizer/test_build_vocab.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import math
import unittest

from typing import Iterable

# self-made modules

from lmp.tokenizer import WhitespaceDictTokenizer


class TestBuildVocab(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.WhitespaceDictTokenizer.build_vocab`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = WhitespaceDictTokenizer()
        self.uncased_tokenizer = WhitespaceDictTokenizer(is_uncased=True)
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
            inspect.signature(WhitespaceDictTokenizer.build_vocab),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='batch_sequences',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[str],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='min_count',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequence(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False,
            object(), lambda x: x, type, None, 0j, 1j,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.build_vocab(batch_sequences=invalid_input)

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`batch_sequences` must be instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_invalid_input_min_count(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, math.inf, b'', 0j, 1j, NotImplemented, ...,
            (), {}, set(), object(), lambda x: x, type, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as cxt_man:
                    tokenizer.build_vocab(
                        batch_sequences=['apple', 'banana'],
                        min_count=invalid_input
                    )

                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`min_count` must be instance of `int`.',
                    msg=msg2
                )

    def test_build_vocabulary_cased_sensitive(self):
        r"""Test whether build vocabulary correctly."""
        msg = 'Inconsistent error message.'
        examples = (
            (
                [
                    'Hello world',
                    'hello world'
                ],
                3 + 4
            ),
        )

        for batch_sequences, vocab_size in examples:
            self.cased_tokenizer.build_vocab(batch_sequences=batch_sequences)
            self.assertEqual(
                self.cased_tokenizer.vocab_size,
                vocab_size,
                msg=msg
            )

    def test_build_vocabulary_cased_insensitive(self):
        r"""Test whether build vocabulary correctly."""
        msg = 'Inconsistent error message.'
        examples = (
            (
                [
                    'Hello world',
                    'hello world'
                ],
                2 + 4
            ),
        )

        for batch_sequences, vocab_size in examples:
            self.uncased_tokenizer.build_vocab(batch_sequences=batch_sequences)
            self.assertEqual(
                self.uncased_tokenizer.vocab_size,
                vocab_size,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
