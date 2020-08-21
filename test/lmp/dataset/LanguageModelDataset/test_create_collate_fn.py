r"""Test `lmp.dataset.LanguageModelDataset.create_collate_fn`.

Usage:
    python -m unittest test/lmp/dataset/LanguageModelDataset/test_create_collate_fn.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

from typing import Callable
from typing import Iterable
from typing import Tuple

# 3rd modules

import torch

# self-made modules

from lmp.dataset._language_model_dataset import LanguageModelDataset
from lmp.tokenizer import BaseTokenizer
from lmp.tokenizer import CharDictTokenizer
from lmp.tokenizer import CharListTokenizer
from lmp.tokenizer import WhitespaceDictTokenizer
from lmp.tokenizer import WhitespaceListTokenizer


class TestCreateCollateFn(unittest.TestCase):
    r"""Test case for `lmp.dataset.LanguageModelDataset.create_collate_fn`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(LanguageModelDataset.create_collate_fn),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=BaseTokenizer,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=-1
                    ),
                ],
                return_annotation=Callable[
                    [Iterable[str]],
                    Tuple[torch.Tensor, torch.Tensor]
                ]
            ),
            msg=msg
        )

    def test_invalid_input_tokenizer(self):
        r"""Raise when input `tokenizer` is invalid."""
        msg1 = 'Must raise `TypeError` when input `tokenizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                LanguageModelDataset([]).create_collate_fn(
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of `lmp.tokenizer.BaseTokenizer`.',
                msg=msg2)

    def test_invalid_input_max_seq_len(self):
        r"""Raise exception when input `max_seq_len` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `max_seq_len` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -2, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...,
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as cxt_man:
                LanguageModelDataset([]).create_collate_fn(
                    tokenizer=CharDictTokenizer(),
                    max_seq_len=invalid_input
                )

            if isinstance(cxt_man.exception, TypeError):
                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`max_seq_len` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    cxt_man.exception.args[0],
                    '`max_seq_len` must be greater than `1` or equal to '
                    '`-1`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `collate_fn`."""
        msg = 'Must return `collate_fn`.'
        examples = (
            CharDictTokenizer,
            CharListTokenizer,
            WhitespaceDictTokenizer,
            WhitespaceListTokenizer,
        )

        for tokenizer_class in examples:
            collate_fn = LanguageModelDataset([]).create_collate_fn(
                tokenizer=tokenizer_class()
            )
            self.assertTrue(inspect.isfunction(collate_fn))
            self.assertEqual(
                inspect.signature(collate_fn),
                inspect.Signature(
                    parameters=[
                        inspect.Parameter(
                            name='batch_sequences',
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=Iterable[str],
                            default=inspect.Parameter.empty
                        ),
                    ],
                    return_annotation=Tuple[torch.Tensor, torch.Tensor]
                ),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
