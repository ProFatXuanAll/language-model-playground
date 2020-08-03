r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.detokenize`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_dict_tokenizer/test_detokenize.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import gc
import math
import unittest

from typing import Iterable

# self-made modules

from lmp.tokenizer import WhitespaceDictTokenizer


class TestDetokenize(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.WhitespaceDictTokenizer.detokenize`."""

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
            inspect.signature(WhitespaceDictTokenizer.detokenize),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokens',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[str],
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=str
            ),
            msg=msg
        )

    def test_invalid_input(self):
        r"""Raise `TypeError` when input is invalid."""
        msg1 = 'Must raise `TypeError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, True, False,
            (1, 2, 3), [1, 2, 3], {1, 2, 3}, None,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                    tokenizer.detokenize(invalid_input)

                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokens` must be instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_expected_return(self):
        r"""Return expected strings."""
        msg = 'Inconsistent detokenization result.'
        examples = (
            (
                ['Hello', 'world!'],
                'Hello world!'
            ),
            (
                ['Hello', 'world', '!'],
                'Hello world !'
            ),
            (
                [],
                ''
            )
        )

        for tokens, ans_sequence in examples:
            for tokenizer in self.tokenizers:
                out_sequence = tokenizer.detokenize(tokens)
                self.assertIsInstance(out_sequence, str, msg=msg)
                self.assertEqual(out_sequence, ans_sequence, msg=msg)

    def test_case_insensitive(self):
        r"""Detokenize does not consider cases."""
        msg = 'Inconsistent detokenization result.'
        examples = (
            ['HeLlo', 'WoRlD', '!'],
            ['hello', 'world', '!'],
        )

        for tokens in examples:
            self.assertEqual(
                self.cased_tokenizer.detokenize(tokens),
                self.uncased_tokenizer.detokenize(tokens),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
