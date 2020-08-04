r"""Test `lmp.tokenizer.BaseDictTokenizer.detokenize`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/test_detokenize.py
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
from typing import List

# self-made modules

from lmp.tokenizer import BaseDictTokenizer


class TestDetokenize(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.BaseDictTokenizer.detokenize`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseDictTokenizer.detokenize),
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
                    ),
                ],
                return_annotation=str
            ),
            msg=msg
        )

    def test_abstract_method(self):
        r"""Raise `NotImplementedError` when subclass did not implement."""
        msg1 = (
            'Must raise `NotImplementedError` when subclass did not '
            'implement.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (True, False)

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseDictTokenizer):
            r"""Intented to not implement `detokenize`."""

            def reset_vocab(self):
                pass
        # pylint: enable=W0231
        # pylint: enable=W0223

        for is_uncased in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                SubClassTokenizer(is_uncased=is_uncased).detokenize('')

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `SubClassTokenizer`: '
                'function `detokenize` not implemented yet.',
                msg=msg2
            )

if __name__ == '__main__':
    unittest.main()
