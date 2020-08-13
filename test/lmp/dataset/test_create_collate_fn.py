r"""Test `lmp.dataset.BaseDataset.crate_collate_fn`.

Usage:
    python -m unittest test/lmp/dataset/test_crate_collate_fn.py
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

from typing import Tuple

# 3rd modules

import torch

# self-made modules

from lmp.dataset import BaseDataset
import lmp.dataset
import lmp.tokenizer

CollateFnReturn = Tuple[
    torch.Tensor,
    torch.Tensor
]


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.crate_collate_fn`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(BaseDataset.create_collate_fn),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=-1
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input_tokenizer(self):
        r"""Raise when `tokenizer` is invalid."""
        msg1 = 'Must raise `TypeError` when `tokenizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            True, False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                BaseDataset(['a', 'b']).create_collate_fn(
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be instance of `lmp.tokenizer.BaseTokenizer`.',
                msg=msg2
            )

    def test_invalid_input_max_seq_len(self):
        r"""Raise when `max_seq_len` is invalid."""
        msg1 = 'Must raise `TypeError` when `max_seq_len` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                BaseDataset(['a', 'b']).create_collate_fn(
                    tokenizer=lmp.tokenizer.CharDictTokenizer(),
                    max_seq_len=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`max_seq_len` must be instance of `int`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `collate_fn`."""
        msg = 'Must return callable function `collate_fn`.'

        collate_fn = BaseDataset.create_collate_fn(
            tokenizer=lmp.tokenizer.CharDictTokenizer(),
            max_seq_len=10
        )
        self.assertTrue(callable(collate_fn), msg=msg)


if __name__ == '__main__':
    unittest.main()
