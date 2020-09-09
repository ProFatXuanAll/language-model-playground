r"""Test `lmp.util._dataset._preprocess_news_collection`.

Usage:
    python -m unittest test.lmp.util._dataset.test_preprocess_news_collection
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

# self-made modules

import lmp.util


class TestPreprocessNewsCollection(unittest.TestCase):
    r"""Test case of `lmp.util._dataset._preprocess_news_collection`"""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'
        self.assertEqual(
            inspect.signature(lmp.util._dataset._preprocess_news_collection),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='column',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.dataset.LanguageModelDataset
            ),
            msg=msg
        )

    def test_invaild_input_column(self):
        r"""Raise `TypeError` or `KeyError` when input `column` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `KeyError` when input `column` is '
            'invaild.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ..., 'invaild key'
        )

        for invaild_input in examples:
            with self.assertRaises((KeyError, TypeError), msg=msg1) as ctx_man:
                lmp.util._dataset._preprocess_news_collection(invaild_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`column` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, KeyError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`column` is not available.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `lmp.dataset.LanguageModelDataset`"""
        msg = 'Must return `lmp.dataset.LanguageModelDataset`.'
        column_parameter = ('desc', 'title')

        for column in column_parameter:
            dataset = lmp.util._dataset._preprocess_news_collection(column)
            self.assertIsInstance(
                dataset,
                lmp.dataset.LanguageModelDataset,
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
