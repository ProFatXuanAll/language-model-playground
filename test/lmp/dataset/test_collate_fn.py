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

class TestInit(unittest.TestCase):
    r"""Test case for `lmp.dataset.BaseDataset.create_collate_fn`'s return function."""

    @classmethod
    def setUpClass(cls):
        cls.is_uncased_range = [True, False]
        cls.max_seq_len_range = list(range(10))
        cls.tokenizer_classes_range = [
            lmp.tokenizer.CharDictTokenizer,
            lmp.tokenizer.CharListTokenizer,
            lmp.tokenizer.WhitespaceDictTokenizer,
            lmp.tokenizer.WhitespaceListTokenizer,
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.is_uncased_range
        del cls.max_seq_len_range
        del cls.tokenizer_classes_range
        gc.collect()

    def setUp(self):
        r"""Setup `collate_fn` instances."""
        self.collate_fn_objs = []

        for is_uncased in self.__class__.is_uncased_range:
            for max_seq_len in self.__class__.max_seq_len_range:
                for tokenizer_class in self.__class__.tokenizer_classes_range:
                    self.collate_fn_objs.append({
                        'collate_fn': BaseDataset.create_collate_fn(
                            tokenizer=tokenizer_class(is_uncased=is_uncased),
                            max_seq_len=max_seq_len
                        ),
                        'is_uncased': is_uncased,
                        'max_seq_len': max_seq_len,
                        'tokeizer_class': tokenizer_class,
                    })

    def tearDown(self):
        r"""Delete `collate_fn` instances."""
        del self.collate_fn_objs
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct function signature.'

        for collate_fn_obj in self.collate_fn_objs:
            self.assertEqual(
                inspect.signature(collate_fn_obj['collate_fn']),
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

    def test_invalid_input_batch_sequences(self):
        r"""Raise when `batch_sequences` is invalid."""
        msg1 = 'Must raise `TypeError` when `batch_sequences` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            for collate_fn_obj in self.collate_fn_objs:
                with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                    collate_fn_obj['collate_fn'](batch_sequences=invalid_input)

                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`batch_sequences` must be instance of `Iterable[str]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `Tuple[torch.Tensor, torch.Tensor]`."""
        msg = (
            'Must return `Tuple[torch.Tensor, torch.Tensor]` with numeric '
            'type `torch.int64`.'
        )
        examples = (
            [
                'Hello',
                'World',
                'Hello World',
            ],
            [
                'Mario use Kimura Lock on Luigi, and Luigi tap out.',
                'Mario use Superman Punch.',
                'Luigi get TKO.',
                'Toad and Toadette are fightting over mushroom (weed).',
            ],
        )

        for batch_sequences in examples:
            for collate_fn_obj in self.collate_fn_objs:
                results = collate_fn_obj['collate_fn'](batch_sequences)
                self.assertIsInstance(results, tuple, msg=msg)
                self.assertEqual(len(results), 2, msg=msg)
                for result in results:
                    self.assertIsInstance(result, torch.Tensor, msg=msg)
                    self.assertEqual(result.dtype, torch.int64, msg=msg)


    def test_return_tensor_size(self):
        r"""Return tensors last dimension must have size `max_seq_len`."""
        msg = 'Return tensors last dimension must have size `max_seq_len`.'
        examples = (
            [
                'Hello',
                'World',
                'Hello World',
            ],
            [
                'Mario use Kimura Lock on Luigi, and Luigi tap out.',
                'Mario use Superman Punch.',
                'Luigi get TKO.',
                'Toad and Toadette are fightting over mushroom (weed).',
            ],
        )

        for batch_sequences in examples:
            for collate_fn_obj in self.collate_fn_objs:
                results = collate_fn_obj['collate_fn'](batch_sequences)
                for result in results:
                    self.assertEqual(
                        collate_fn_obj['max_seq_len'],
                        result.size(-1),
                        msg=msg
                    )

if __name__ == '__main__':
    unittest.main()
