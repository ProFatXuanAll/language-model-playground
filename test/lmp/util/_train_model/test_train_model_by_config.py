r"""Test `lmp.util.train_model_by_config.`.

Usage:
    python -m unittest test.lmp.util._train_model.test_train_model_by_config
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


class TestTrainModelByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.train_model_by_config`."""

    def setUp(self):
        r"""Set up parameters for `train_model_by_config`."""
        my_data = ['apple', 'banana', 'papaya']
        tokenizer = lmp.tokenizer.CharDictTokenizer()

        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            dataset='my_data',
            experiment='util_train_model_by_config_unittest',
            model_class='rnn',
            optimizer_class='sgd',
            tokenizer_class='char_dict'
        )
        self.dataset = lmp.dataset.BaseDataset(my_data)
        self.model = lmp.model.BaseRNNModel(
            d_emb=4,
            d_hid=4,
            dropout=0.2,
            num_rnn_layers=1,
            num_linear_layers=1,
            pad_token_id=0,
            vocab_size=tokenizer.vocab_size
        )
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.5
        )
        self.tokenizer = tokenizer

    def tearDown(self):
        r"""Delete parameters for `train_model_by_config`."""
        del self.checkpoint
        del self.config
        del self.dataset
        del self.model
        del self.optimizer
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.train_model_by_config),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
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
                        name='model',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Union[
                            lmp.model.BaseRNNModel,
                            lmp.model.BaseResRNNModel
                        ],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='optimizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Union[
                           torch.optim.SGD,
                           torch.optim.Adam,
                        ],
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

    def test_invalid_input_checkpoint(self):
        r"""Raise when `checkpoint` is invalid."""
        msg1 = 'Must raise `TypeError` when `checkpoint` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model_by_config(
                    checkpoint=invalid_input,
                    config=self.config,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
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
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=self.optimizer,
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
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=invalid_input,
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must be an instance of '
                    '`lmp.dataset.BaseDataset`.',
                    msg=msg2
                )

    def test_invalid_input_model(self):
        r"""Raise when `model` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `model` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=self.dataset,
                    model=invalid_input,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model` must be an instance of '
                    '`Union['
                        'lmp.model.BaseRNNModel,'
                        'lmp.model.BaseResRNNModel'
                    ']`.',
                    msg=msg2
                )
 
    def test_invalid_input_optimizer(self):
        r"""Raise when `optimizer` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `optimizer` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=invalid_input,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`optimizer` must be an instance of '
                    '`Union['
                        'torch.optim.SGD,'
                        'torch.optim.Adam'
                    ']`.',
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
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=self.optimizer,
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
