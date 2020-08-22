r"""Test `lmp.tokenizer.BaseTokenizer.convert_tokens_to_ids`.

Usage:
    python -m unittest \
        test.lmp.tokenizer._base_tokenizer.test_convert_tokens_to_ids
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

from typing import Iterable
from typing import List

# self-made modules

from lmp.tokenizer import BaseTokenizer


class TestConvertTokensToIds(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.convert_tokens_to_ids`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseTokenizer.convert_tokens_to_ids),
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
                return_annotation=List[int]
            ),
            msg=msg
        )

    def test_abstract_method(self):
        r"""Raise `NotImplementedError` when subclass did not implement."""
        msg1 = (
            'Must raise `NotImplementedError` when subclass did not implement.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (True, False)

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseTokenizer):
            r"""Intented to not implement `convert_token_to_id`."""

            def reset_vocab(self):
                pass
        # pylint: enable=W0231
        # pylint: enable=W0223

        for is_uncased in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                SubClassTokenizer(
                    is_uncased=is_uncased
                ).convert_tokens_to_ids([''])

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `SubClassTokenizer`: '
                'method `convert_token_to_id` not implemented yet.',
                msg=msg2
            )


if __name__ == '__main__':
    unittest.main()
