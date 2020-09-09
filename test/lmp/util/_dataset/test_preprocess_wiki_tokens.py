r"""Test `lmp.util._dataset._preprocess_wiki_tokens`.

Usage:
    python -m unittest test.lmp.util._dataset.test_preprocess_wiki_tokens
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import os
import unittest

# self-made modules

import lmp.path
import lmp.util


class TestPreprocessWikiTokens(unittest.TestCase):
    r"""Test case of `lmp.util._dataset._preprocess_news_collection`"""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'
        self.assertEqual(
            inspect.signature(lmp.util._dataset._preprocess_wiki_tokens),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='split',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.dataset.LanguageModelDataset
            ),
            msg=msg
        )

    def test_invaild_input_split(self):
        r"""Raise `TypeError` or `FileNotFoundError` when input `split` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `FileNotFoundError` when input `split` '
            'is invaild.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ..., 'NotExistFile'
        )

        for invaild_input in examples:
            with self.assertRaises((FileNotFoundError, TypeError), msg=msg1) as ctx_man:
                lmp.util._dataset._preprocess_wiki_tokens(invaild_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`split` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, FileNotFoundError):
                file_path = os.path.join(
                    f'{lmp.path.DATA_PATH}',
                    f'wiki.{invaild_input}.tokens'
                )
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'file {file_path} does not exist.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `lmp.dataset.LanguageModelDataset`"""
        msg = 'Must return `lmp.dataset.LanguageModelDataset`.'
        split_parameter = ('train', 'valid','test')

        for split in split_parameter:
            dataset = lmp.util._dataset._preprocess_wiki_tokens(split)
            self.assertIsInstance(
                dataset,
                lmp.dataset.LanguageModelDataset,
                msg=msg
            )

if __name__ == '__main__':
    unittest.main()