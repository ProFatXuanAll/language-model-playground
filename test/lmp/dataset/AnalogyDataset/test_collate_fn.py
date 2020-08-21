r"""Test `lmp.dataset.AnalogyDataset.collate_fn`.

Usage:
    python -m unittest test/lmp/dataset/AnalogyDataset/test_collate_fn.py
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
from typing import List
from typing import Tuple

# 3rd modules

import torch

# self-made modules

from lmp.dataset._analogy_dataset import AnalogyDataset
from lmp.tokenizer import CharDictTokenizer
from lmp.tokenizer import CharListTokenizer
from lmp.tokenizer import WhitespaceDictTokenizer
from lmp.tokenizer import WhitespaceListTokenizer


class TestCollateFn(unittest.TestCase):
    r"""Test case for `lmp.dataset.AnalogyDataset.collate_fn`."""

    @classmethod
    def setUpClass(cls):
        cls.is_uncased_range = [True, False]
        cls.tokenizer_class_range = [
            CharDictTokenizer,
            CharListTokenizer,
            WhitespaceDictTokenizer,
            WhitespaceListTokenizer,
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.is_uncased_range
        del cls.tokenizer_class_range
        gc.collect()

    def setUp(self):
        r"""Setup `collate_fn` instances."""
        self.collate_fn_objs = []

        cls = self.__class__
        for is_uncased in cls.is_uncased_range:
            for tokenizer_class in cls.tokenizer_class_range:
                self.collate_fn_objs.append({
                    'collate_fn': AnalogyDataset.create_collate_fn(
                        tokenizer=tokenizer_class(is_uncased=is_uncased)
                    ),
                    'is_uncased': is_uncased,
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

    def test_invalid_input_batch_analogy(self):
        r"""Raise exception when input `batch_analogy` is invalid."""
        msg1 = (
            'Must raise `TypeError` , `IndexError` or `ValueError` '
            'when input `batch_analogy` is invalid.'
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
                        (TypeError, ValueError, IndexError),
                        msg=msg1
                ) as ctx_man:
                    collate_fn_obj['collate_fn'](batch_analogy=invalid_input)

                if isinstance(ctx_man.exception, TypeError):
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_analogy` must be an instance of '
                        '`Iterable[Iterable[str]]`.',
                        msg=msg2
                    )
                elif isinstance(ctx_man.exception, IndexError):
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_analogy` must be size (batch_size,5)',
                        msg=msg2
                    )
                else:
                    self.assertEqual(
                        ctx_man.exception.args[0],
                        '`batch_analogy` must not be empty.',
                        msg=msg2
                    )

    def test_return_type(self):
        r"""Return 
        `Tuple[torch.Tensor,torch.Tensor,torch.Tensor,str,str,]`."""
        msg = (
            'Must return '
            '`Tuple[torch.Tensor,torch.Tensor,torch.Tensor,str,str,]` '
            'with Tensor have numeric type `torch.int64`.')
        examples = (
            [
                [
                    'Taiwan',
                    'Taipei',
                    'Japan',
                    'Tokyo',
                    'capital',
                ],
                [
                    'write',
                    'writes',
                    'sad',
                    'sads',
                    'grammer',
                ],
            ],
            [
                [
                    'write',
                    'writes',
                    'sad',
                    'sads',
                    'grammer',
                ]
            ],
        )

        for batch_analogy in examples:
            for collate_fn_obj in self.collate_fn_objs:
                results = collate_fn_obj['collate_fn'](batch_analogy)
                self.assertIsInstance(results, tuple, msg=msg)
                self.assertEqual(len(results), 5, msg=msg)
                for result in results[0:3]:
                    self.assertIsInstance(result, torch.Tensor, msg=msg)
                    self.assertEqual(result.dtype, torch.int64, msg=msg)
                for result in results[3:5]:
                    self.assertIsInstance(result, Iterable, msg=msg)
                    for word in result:
                        self.assertIsInstance(word, str, msg=msg)

    def test_return_value(self):
        r"""Return value must have three word id and two string.
        Those two strings must be the same as the example.
        """
        msg = (
            'Return value must have three word id and two string.'
            'Those two strings must be the same as the example.'
        )
        examples = (
            [
                [
                    'Taiwan',
                    'Taipei',
                    'Japan',
                    'Tokyo',
                    'capital',
                ]
            ],
            [
                [
                    'write',
                    'writes',
                    'sad',
                    'sads',
                    'grammer',
                ]
            ],
        )

        for batch_analogy in examples:
            for collate_fn_obj in self.collate_fn_objs:
                _, _, _, word_d, category = collate_fn_obj['collate_fn'](
                    batch_analogy
                )
                self.assertEqual(word_d[0], batch_analogy[0][3], msg=msg)
                self.assertEqual(category[0], batch_analogy[0][4], msg=msg)


if __name__ == '__main__':
    unittest.main()
