r"""Test `lmp.util.train_tokenizer_by_config.`.

Usage:
    python -m unittest test.lmp.util._train_tokenizer.test_train_tokenizer_by_config
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

from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestTrainTokenizerByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.train_tokenizer_by_config`."""

    def setUp(self):
        r"""Set up parameters for `train_tokenizer_by_config`."""
        my_data = ['apple', 'banana', 'papaya']
        self.config = lmp.config.BaseConfig(
            dataset='my_data',
            experiment='util_train_tokenizer_by_config_unittet',
            tokenizer_class='char_dict'
        )
        self.dataset = lmp.dataset.BaseDataset(my_data)
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete parameters for `train_tokenizer_by_config`."""
        del self.config
        del self.dataset
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.train_tokenizer_by_config),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='config',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.config.BaseConfig,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.dataset.BaseDataset,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=inspect.Parameter.empty
            ),
            msg=msg
        )

    def test_invalid_input_config(self):
        r"""Raise when `config` is invalid."""
        msg1 = 'Must raise `TypeError` when `config` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_tokenizer_by_config(
                    config=invalid_input,
                    dataset=self.dataset,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`config` must be an instance of `lmp.config.BaseConfig`.',
                    msg=msg2
                )

    def test_invalid_input_dataset(self):
        r"""Raise when `dataset` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `dataset` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_tokenizer_by_config(
                    config=self.config,
                    dataset=invalid_input,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must be an instance of '
                    '`lmp.dataset.BaseDataset`.',
                    msg=msg2
                )

    def test_invalid_input_tokenizer(self):
        r"""Raise when `tokenizer` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `tokenizer` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_tokenizer_by_config(
                    config=self.config,
                    dataset=self.dataset,
                    tokenizer=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokenizer` must be an instance of '
                    '`lmp.tokenizer.BaseTokenizer`.',
                    msg=msg2
                )

if __name__ == '__main__':
    unittest.main()
