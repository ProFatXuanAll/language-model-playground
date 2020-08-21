r"""Test `lmp.dataset.AnalogyDataset.create_collate_fn`.

Usage:
    python -m unittest test/lmp/dataset/AnalogyDataset/test_create_collate_fn.py
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
from typing import List
from typing import Tuple

# 3rd modules

import torch

# self-made modules

from lmp.dataset import AnalogyDataset
from lmp.tokenizer import BaseTokenizer
from lmp.tokenizer import CharDictTokenizer
from lmp.tokenizer import CharListTokenizer
from lmp.tokenizer import WhitespaceDictTokenizer
from lmp.tokenizer import WhitespaceListTokenizer


class TestCreateCollateFn(unittest.TestCase):
    r"""Test case for `lmp.dataset.AnalogyDataset.create_collate_fn`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(AnalogyDataset.create_collate_fn),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=BaseTokenizer,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=Callable[
                    [List[List[str]]],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        str,
                        str,
                    ],
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
                AnalogyDataset(
                    []).create_collate_fn(
                    tokenizer=invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
                msg=msg2)

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
            collate_fn = AnalogyDataset([]).create_collate_fn(
                tokenizer=tokenizer_class()
            )
            self.assertTrue(inspect.isfunction(collate_fn))
            self.assertEqual(
                inspect.signature(collate_fn),
                inspect.Signature(
                    parameters=[
                        inspect.Parameter(
                            name='batch_analogy',
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=Iterable[Iterable[str]],
                            default=inspect.Parameter.empty
                        ),
                    ],
                    return_annotation=Tuple[
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        str,
                                        str,
                                    ]
                ),
                msg=msg
            )


if __name__ == '__main__':
    unittest.main()
