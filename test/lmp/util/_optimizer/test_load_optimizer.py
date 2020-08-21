r"""Test `lmp.util.load_optimizer.`.

Usage:
    python -m unittest test.lmp.util._optimizer.test_load_optimizer
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
import lmp.path


class TestLoadOptimizer(unittest.TestCase):
    r"""Test Case for `lmp.util.load_optimizer`."""

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
        cls.model_param_values = [v for v in cls.model_parameters.values()]
        cls.learning_rate_range = [0.0, 0.1, 0.5, 0.9]
        cls.optimizer_class_range = ['sgd', 'adam']

    @classmethod
    def tearDownClass(cls):
        del cls.learning_rate_range
        del cls.model_param_values
        del cls.model_parameters
        del cls.optimizer_class_range
        gc.collect()

    def setUp(self):
        r"""Set up parameters for `load_optimizer`."""
        self.checkpoint = -1
        self.experiment = 'util_load_optimizer_unittest'
        self.learning_rate = 0.25
        self.optimizer_class = 'sgd'
        self.parameters = lmp.model.BaseRNNModel(
            d_emb=4,
            d_hid=4,
            dropout=0.2,
            num_rnn_layers=1,
            num_linear_layers=1,
            pad_token_id=0,
            vocab_size=10
        ).parameters()

        cls = self.__class__
        self.parameters_obj = []
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
            self.parameters_obj.append(list(model.parameters()))

    def tearDown(self):
        r"""Delete parameters for `load_optimizer`."""
        del self.checkpoint
        del self.experiment
        del self.learning_rate
        del self.optimizer_class
        del self.parameters
        del self.parameters_obj
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_optimizer),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='learning_rate',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='optimizer_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='parameters',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Iterator[torch.nn.Parameter],
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
                lmp.util.load_optimizer(
                    checkpoint=invalid_input,
                    experiment=self.experiment,
                    learning_rate=self.learning_rate,
                    optimizer_class=self.optimizer_class,
                    parameters=self.parameters
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
                )

    def test_invalid_input_experiment(self):
        r"""Raise when `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `experiment` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_optimizer(
                    checkpoint=self.checkpoint,
                    experiment=invalid_input,
                    learning_rate=self.learning_rate,
                    optimizer_class=self.optimizer_class,
                    parameters=self.parameters
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must be an instance of `str`.',
                    msg=msg2
                )

    def test_invalid_input_learning_rate(self):
        r"""Raise when `learning_rate` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `learning_rate` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0j, 1j, '', b'', [], (), {}, set(),
            object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_optimizer(
                    checkpoint=self.checkpoint,
                    experiment=self.experiment,
                    learning_rate=invalid_input,
                    optimizer_class=self.optimizer_class,
                    parameters=self.parameters
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`learning_rate` must be an instance of `float`.',
                    msg=msg2
                )

    def test_invalid_input_optimizer_class(self):
        r"""Raise when `optimizer_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `optimizer_class` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                (TypeError, ValueError),
                msg=msg1
            ) as ctx_man:
                lmp.util.load_optimizer(
                    checkpoint=self.checkpoint,
                    experiment=self.experiment,
                    learning_rate=self.learning_rate,
                    optimizer_class=invalid_input,
                    parameters=self.parameters
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`optimizer_class` must be an instance of `str`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'`{invalid_input}` does not support\n'
                    'Supported options:' +
                    ''.join(list(map(
                        lambda option: f'\n\t--optimizer {option}',
                        [
                            'sgd',
                            'adam',
                        ]
                    ))),
                    msg=msg2
                )

    def test_invalid_input_parameters(self):
        r"""Raise when `parameters` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `parameters` '
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
                lmp.util.load_optimizer(
                    checkpoint=self.checkpoint,
                    experiment=self.experiment,
                    learning_rate=self.learning_rate,
                    optimizer_class=self.optimizer_class,
                    parameters=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`parameters` must be an instance of '
                    '`Iterator[torch.nn.Parameter]`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `torch.optim.SGD` or `torch.optim.Adam`."""
        msg = (
            'Must return `torch.optim.SGD` or `torch.optim.Adam`.'
        )
        examples = (
            (
                -1,
                'util_load_optimizer_unittest',
                0.25,
                'sgd',
                lmp.model.BaseRNNModel(
                    d_emb=4,
                    d_hid=4,
                    dropout=0.2,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ).parameters(),
            ),
            (
                -1,
                'util_load_optimizer_unittest',
                0.32,
                'adam',
                lmp.model.BaseResRNNModel(
                    d_emb=10,
                    d_hid=5,
                    dropout=0.15,
                    num_rnn_layers=2,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=30
                ).parameters(),
            ),
        )
        examples = (
            (
                learning_rate,
                optimizer_class,
                parameters,
            )
            for learning_rate in self.__class__.learning_rate_range
            for optimizer_class in self.__class__.optimizer_class_range
            for parameters in self.parameters_obj
        )

        for learning_rate, optimizer_class, parameters in examples:
            optimizer = lmp.util.load_optimizer(
                checkpoint=-1,
                experiment='news_collection',
                learning_rate=learning_rate,
                optimizer_class=optimizer_class,
                parameters=parameters
            )

            try:
                self.assertIsInstance(optimizer, torch.optim.SGD, msg=msg)
            except AssertionError:
                self.assertIsInstance(optimizer, torch.optim.Adam, msg=msg)


if __name__ == '__main__':
    unittest.main()
