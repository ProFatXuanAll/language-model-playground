r"""Test `lmp.util.train_tokenizer.`.

Usage:
    python -m unittest test.lmp.util._train_tokenizer.test_train_tokenizer
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

from itertools import product

# self-made modules

import lmp.dataset
import lmp.tokenizer
import lmp.util


class TestTrainTokenizer(unittest.TestCase):
    r"""Test case for `lmp.util.train_tokenizer`."""

    @classmethod
    def setUpClass(cls):
        r"""Setup dynamic parameters."""
        cls.dataset = 'I-AM-A-TEST-DATASET'
        cls.experiment = 'I-AM-A-TEST-EXPERIMENT'
        cls.tokenizer_parameters = {
            'is_uncased': [False, True],
            'batch_sequences': [
                ['hello', 'hello world'],
                ['world', 'hello world'],
            ],
            'min_count': [1, 2],
            'tokenizer_cstr': [
                lmp.tokenizer.CharDictTokenizer,
                lmp.tokenizer.CharListTokenizer,
                lmp.tokenizer.WhitespaceDictTokenizer,
                lmp.tokenizer.WhitespaceListTokenizer,
                lmp.tokenizer.CharDictTokenizer,
                lmp.tokenizer.CharListTokenizer,
            ],
        }

    @classmethod
    def tearDownClass(cls):
        r"""Delete dynamic parameters."""
        del cls.dataset
        del cls.experiment
        del cls.tokenizer_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.dataset = lmp.dataset.BaseDataset([''])
        self.min_count = 1
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.dataset
        del self.min_count
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.train_tokenizer),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.dataset.BaseDataset,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='min_count',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_invalid_input_dataset(self):
        r"""Raise `TypeError` when input `dataset` is invalid."""
        msg1 = 'Must raise `TypeError` when input `dataset` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_tokenizer(
                    dataset=invalid_input,
                    min_count=self.min_count,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`dataset` must be an instance of `lmp.dataset.BaseDataset`.',
                msg=msg2
            )

    def test_invalid_input_min_count(self):
        r"""Raise exception when input `min_count` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `min_count` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', (), [], {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.train_tokenizer(
                    dataset=self.dataset,
                    min_count=invalid_input,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`min_count` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`min_count` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_tokenizer(self):
        r"""Raise `TypeError` when input `tokenizer` is invalid."""
        msg1 = 'Must raise `TypeError` when input `tokenizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_tokenizer(
                    dataset=self.dataset,
                    min_count=self.min_count,
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
                msg=msg2
            )

    def test_increase_vocab(self):
        r"""Increase vocabulary."""
        msg = 'Must increase vocabulary.'

        for (
                is_uncased,
                batch_sequences,
                min_count,
                tokenizer_cstr
        ) in product(*self.__class__.tokenizer_parameters.values()):
            dataset = lmp.dataset.BaseDataset(batch_sequences=batch_sequences)
            tokenizer = tokenizer_cstr(is_uncased=is_uncased)
            v1 = tokenizer.vocab_size

            lmp.util.train_tokenizer(
                dataset=dataset,
                min_count=min_count,
                tokenizer=tokenizer
            )

            self.assertGreater(tokenizer.vocab_size, v1, msg=msg)


if __name__ == '__main__':
    unittest.main()
