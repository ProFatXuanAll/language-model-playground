r"""Test `lmp.dataset.LanguageModelDataset.collate_fn`.

Usage:
    python -m unittest test.lmp.dataset.LanguageModelDataset.test_collate_fn
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

from lmp.dataset._language_model_dataset import LanguageModelDataset
from lmp.tokenizer import CharDictTokenizer
from lmp.tokenizer import CharListTokenizer
from lmp.tokenizer import WhitespaceDictTokenizer
from lmp.tokenizer import WhitespaceListTokenizer


class TestCollateFn(unittest.TestCase):
    r"""Test case for `lmp.dataset.LanguageModelDataset.collate_fn`."""

    @classmethod
    def setUpClass(cls):
        cls.is_uncased_range = [True, False]
        cls.max_seq_len_range = [-1] + list(range(2, 10))
        cls.tokenizer_class_range = [
            CharDictTokenizer,
            CharListTokenizer,
            WhitespaceDictTokenizer,
            WhitespaceListTokenizer,
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.is_uncased_range
        del cls.max_seq_len_range
        del cls.tokenizer_class_range
        gc.collect()

    def setUp(self):
        r"""Setup `collate_fn` instances."""
        self.collate_fn_objs = []

        cls = self.__class__
        for is_uncased in cls.is_uncased_range:
            for max_seq_len in cls.max_seq_len_range:
                for tokenizer_class in cls.tokenizer_class_range:
                    self.collate_fn_objs.append({
                        'collate_fn': LanguageModelDataset.create_collate_fn(
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
        r"""Raise exception when input `batch_sequences` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`batch_sequences` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ..., [False], [True], [0],
            [1], [-1], [0.0], [1.0], [math.nan], [-math.nan], [math.inf],
            [-math.inf], [0j], [1j], [b''], [()], [[]], [{}], [set()],
            [object()], [lambda x: x], [type], [None], [NotImplemented], [...],
            ['', False], ['', True], ['', 0], ['', 1], ['', -1], ['', 0.0],
            ['', 1.0], ['', math.nan], ['', -math.nan], ['', math.inf],
            ['', -math.inf], ['', 0j], ['', 1j], ['', b''], ['', ()], ['', []],
            ['', {}], ['', set()], ['', object()], ['', lambda x: x],
            ['', type], ['', None], ['', NotImplemented], ['', ...],
        )

        for invalid_input in examples:
            for collate_fn_obj in self.collate_fn_objs:
                with self.assertRaises(
                        (TypeError, ValueError),
                        msg=msg1
                ) as ctx_man:
                    collate_fn_obj['collate_fn'](batch_sequences=invalid_input)

                if isinstance(ctx_man.exception, TypeError):
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_sequences` must be an instance of '
                        '`Iterable[str]`.',
                        msg=msg2
                    )
                else:
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_sequences` must not be empty.',
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
            [''],
        )

        for batch_sequences in examples:
            for collate_fn_obj in self.collate_fn_objs:
                results = collate_fn_obj['collate_fn'](batch_sequences)
                self.assertIsInstance(results, tuple, msg=msg)
                self.assertEqual(len(results), 2, msg=msg)
                for result in results:
                    self.assertIsInstance(result, torch.Tensor, msg=msg)
                    self.assertEqual(result.dtype, torch.int64, msg=msg)

    def test_return_value(self):
        r"""Return tensors have exact same size and shift one position."""
        msg = (
            'Return tensors must have exact same size and shift one position.'
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
            [''],
        )

        for batch_sequences in examples:
            for collate_fn_obj in self.collate_fn_objs:
                x, y = collate_fn_obj['collate_fn'](batch_sequences)
                self.assertEqual(x.size(), y.size(), msg=msg)
                self.assertEqual(len(x.size()), 2, msg=msg)
                self.assertEqual(x.size(0), len(batch_sequences), msg=msg)

                batch_bool = x[:, 1:] == y[:, :-1]
                for bool_sequence in batch_bool:
                    for each_bool in bool_sequence:
                        self.assertTrue(each_bool.item(), msg=msg)

    def test_truncate_and_pad(self):
        r"""Batch token ids' length must be `max_seq_len - 1`."""
        msg = 'Batch token ids\' length must be `max_seq_len - 1`.'
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
            [''],
        )

        for batch_sequences in examples:
            for collate_fn_obj in self.collate_fn_objs:
                x, y = collate_fn_obj['collate_fn'](batch_sequences)
                max_seq_len = collate_fn_obj['max_seq_len']

                if max_seq_len == -1:
                    continue

                self.assertEqual(x.size(-1), max_seq_len - 1, msg=msg)
                self.assertEqual(y.size(-1), max_seq_len - 1, msg=msg)


if __name__ == '__main__':
    unittest.main()
