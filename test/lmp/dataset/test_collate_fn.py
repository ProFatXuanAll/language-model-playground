r"""Test `lmp.dataset.BaseDataset.collate_fn`.

Usage:
    python -m unittest test/lmp/dataset/test_collate_fn.py
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
from typing import Tuple

# 3rd modules

import torch

# self-made modules

import lmp.tokenizer

from lmp.dataset import BaseDataset

CollateFnReturn = Tuple[
    torch.Tensor,
    torch.Tensor
]


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.create_collate_fn`'s inner method `collat_fn`."""

    def setUp(self):
        r"""Setup `collate_fn` instances."""
        self.collate_fn = BaseDataset.create_collate_fn(
            tokenizer=lmp.tokenizer.CharDictTokenizer(),
            max_seq_len=10
        )

    def tearDown(self):
        r"""Delete `collate_fn` instances."""
        del self.collate_fn
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(self.collate_fn),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='batch_sequences',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterable[str],
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=CollateFnReturn
            ),
            msg=msg
        )

    def test_invalid_input_batch_sequences(self):
        r"""Raise when `batch_sequences` is invalid."""
        msg1 = 'Must raise `TypeError` when `batch_sequences` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            True, False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                self.collate_fn(batch_sequences=invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`batch_sequences` must be instance of `Iterable[str]`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `Tuple(Tensor, Tensor)`."""
        msg = 'Must return `Tuple(Tensor, Tensor)`.'
        examples = (
            [
                'Kimura lock.',
                'Superman punch.'
                'Close Guard.'
            ],
        )

        for batch_sequences in examples:
            data_handled = self.collate_fn(batch_sequences)
            self.assertIsInstance(data_handled, tuple, msg=msg)
            self.assertEqual(len(data_handled), 2, msg=msg)
            for tensor in data_handled:
                self.assertIsInstance(
                    tensor,
                    torch.Tensor,
                    msg=msg
                )
                self.assertEqual(
                    tensor.dtype,
                    torch.int64,
                    msg=msg
                )

    def test_inner_method_return_value(self):
        r"""Return two tensor."""
        msg = 'Inconsistent error message.'
        examples = (
            (
                ['Hello'],
                [0, 3, 3, 3, 3, 3, 1, 2, 2],
                [3, 3, 3, 3, 3, 1, 2, 2, 2],
            ),
            (
                ['Hello from the other side.'],
                [0, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 1],
            ),
        )

        for batch_sequences, ans_x, ans_y in examples:
            data_handled = self.collate_fn(batch_sequences)
            for x, y in (data_handled,):
                self.assertEqual(
                    x.squeeze(0).tolist(),
                    ans_x,
                    msg=msg
                )
                self.assertEqual(
                    y.squeeze(0).tolist(),
                    ans_y,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
