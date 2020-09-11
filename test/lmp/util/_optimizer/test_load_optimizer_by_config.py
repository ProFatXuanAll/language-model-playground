r"""Test `lmp.util.load_optimizer_by_config.`.

Usage:
    python -m unittest test.lmp.util._optimizer.test_load_optimizer_by_config
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
from typing import Iterator
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model


class TestLoadOptimizerByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.load_optimizer_by_config`."""

    @classmethod
    def setUpClass(cls):
        cls.model_parameters = {
            'd_emb': [5, 6],
            'd_hid': [7, 9],
            'dropout': [0.0, 0.1, 0.5, 1.0],
            'num_linear_layers': [3, 6],
            'num_rnn_layers': [2, 5],
            'pad_token_id': [0, 1, 2, 3],
            'vocab_size': [10, 15]
        }
        cls.config_parameters = {
            'learning_rate': [.5, .6],
            'optimizer_class': ['sgd', 'adam']
        }
        cls.model_param_values = [v for v in cls.model_parameters.values()]
        cls.config_param_values = [v for v in cls.config_parameters.values()]

    @classmethod
    def tearDownClass(cls):
        del cls.model_parameters
        del cls.config_parameters
        del cls.model_param_values
        del cls.config_param_values
        gc.collect()

    def setUp(self):
        r"""Set up parameters for `load_optimizer_by_config`."""
        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            dataset='news_collection',
            experiment='util_load_by_config_optimizer_unittest'
        )
        self.model = lmp.model.BaseRNNModel(
            d_emb=4,
            d_hid=4,
            dropout=0.2,
            num_rnn_layers=1,
            num_linear_layers=1,
            pad_token_id=0,
            vocab_size=10
        )

        cls = self.__class__
        self.model_obj = []
        for (
            d_emb,
            d_hid,
            dropout,
            num_linear_layers,
            num_rnn_layers,
            pad_token_id,
            vocab_size
        ) in product(*cls.model_param_values):
            if vocab_size <= pad_token_id:
                continue
            model = lmp.model.BaseRNNModel(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )
            self.model_obj.append(model)

        self.config_obj = []
        for (
            learning_rate,
            optimizer_class
        ) in product(*cls.config_param_values):
            config = lmp.config.BaseConfig(
                dataset='news_collection',
                experiment='util_load_optimizer_by_config_unittest',
                learning_rate=learning_rate,
                optimizer_class=optimizer_class
            )
            self.config_obj.append(config)

    def tearDown(self):
        r"""Delete parameters for `load_optimizer_by_config`."""
        del self.checkpoint
        del self.config
        del self.config_obj
        del self.model
        del self.model_obj
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_optimizer_by_config),
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
                        name='model',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Union[
                            lmp.model.BaseRNNModel,
                            lmp.model.BaseResRNNModel,
                            lmp.model.BaseSelfAttentionRNNModel,
                            lmp.model.BaseSelfAttentionResRNNModel
                        ],
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=Union[
                    torch.optim.SGD,
                    torch.optim.Adam,
                ]
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
                lmp.util.load_optimizer_by_config(
                    checkpoint=invalid_input,
                    config=self.config,
                    model=self.model
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
                )

    def test_invalid_input_config(self):
        r"""Raise when `config` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `config` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_optimizer_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input,
                    model=self.model
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`config` must be an instance of `lmp.config.BaseConfig`.',
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
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_optimizer_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    model=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model` must be an instance of '
                    '`Union[lmp.model.BaseRNNModel, '
                    'lmp.model.BaseResRNNModel, '
                    'lmp.model.BaseSelfAttentionRNNModel, '
                    'lmp.model.BaseSelfAttentionResRNNModel]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `torch.optim.SGD` or `torch.optim.Adam`."""
        msg = (
            'Must return `torch.optim.SGD` or `torch.optim.Adam`.'
        )
        examples = (
            (
                config,
                model,
            )
            for config in self.config_obj
            for model in self.model_obj
        )

        for config, model in examples:
            optimizer = lmp.util.load_optimizer_by_config(
                checkpoint=-1,
                config=config,
                model=model
            )

            try:
                self.assertIsInstance(optimizer, torch.optim.SGD, msg=msg)
            except AssertionError:
                self.assertIsInstance(optimizer, torch.optim.Adam, msg=msg)


if __name__ == '__main__':
    unittest.main()
