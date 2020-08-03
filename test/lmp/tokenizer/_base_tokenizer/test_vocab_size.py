r"""Test `lmp.tokenizer.BaseTokenizer.vocab_size`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_tokenizer/test_vocab_size.py
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
    r"""Test Case for `lmp.tokenizer.BaseTokenizer.vocab_size`."""

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

    def test_expected_return(self):
        r"""Return expected number."""
        msg = 'Inconsistent vocabulary size.'
        examples = (
            (True, []),
            (False, []),
            (True, [1]),
            (False, [1]),
            (True, [1, 2, 3]),
            (False, [1, 2, 3]),
            (True, {}),
            (False, {}),
            (True, {1: 1}),
            (False, {1: 1}),
            (True, {1: 1, 2: 2, 3: 3}),
            (False, {1: 1, 2: 2, 3: 3}),
        )

        for is_uncased, token_to_id in examples:
            # pylint: disable=W0223
            # pylint: disable=W0231
            # pylint: disable=W0640
            class SubClassTokenizer(BaseTokenizer):
                r"""Intented to not implement `vocab_size`."""

                def reset_vocab(self):
                    self.token_to_id = token_to_id
            # pylint: enable=W0640
            # pylint: enable=W0231
            # pylint: enable=W0223

            self.assertEqual(
                SubClassTokenizer(is_uncased=is_uncased).vocab_size,
                len(token_to_id),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
