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
import os
import unittest

from itertools import product
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.config
import lmp.model
import lmp.path
import lmp.util


class TestLoadOptimizerByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.load_optimizer_by_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.checkpoint = 10
        cls.dataset = 'I-AM-TEST-DATASET'
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.model_parameters = {
            'd_emb': [1, 2],
            'd_hid': [1, 2],
            'dropout': [0.0, 0.1],
            'model_cstr': [
                lmp.model.BaseRNNModel,
                lmp.model.GRUModel,
                lmp.model.LSTMModel,
                lmp.model.BaseResRNNModel,
                lmp.model.ResGRUModel,
                lmp.model.ResLSTMModel,
            ],
            'num_linear_layers': [1, 2],
            'num_rnn_layers': [1, 2],
            'pad_token_id': [0, 1],
            'vocab_size': [5, 10]
        }
        cls.optimizer_parameters = {
            'learning_rate': [0.0, 0.1, 0.5, 0.9],
            'optimizer': [
                ('sgd', torch.optim.SGD),
                ('adam', torch.optim.Adam),
            ],
        }
        cls.test_dir = os.path.join(lmp.path.DATA_PATH, cls.experiment)
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory and delete dynamic parameters."""
        os.removedirs(cls.test_dir)
        del cls.checkpoint
        del cls.dataset
        del cls.experiment
        del cls.model_parameters
        del cls.optimizer_parameters
        del cls.test_dir
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            dataset=self.__class__.dataset,
            experiment=self.__class__.experiment
        )
        self.model = lmp.model.BaseRNNModel(
            d_emb=1,
            d_hid=1,
            dropout=0.0,
            num_linear_layers=1,
            num_rnn_layers=1,
            pad_token_id=0,
            vocab_size=5
        )

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.checkpoint
        del self.config
        del self.model
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
                            lmp.model.BaseResRNNModel
                        ],
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=Union[torch.optim.SGD, torch.optim.Adam]
            ),
            msg=msg
        )

    def test_invalid_input_checkpoint(self):
        r"""Raise exception when input `checkpoint` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `checkpoint` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -2, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', (), [], {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
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
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be bigger than or equal to `-1`.',
                    msg=msg2
                )

    def test_invalid_input_config(self):
        r"""Raise `TypeError` when input `config` is invalid."""
        msg1 = 'Must raise `TypeError` when input `config` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_optimizer_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input,
                    model=self.model
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`config` must be an instance of `lmp.config.BaseConfig`.',
                msg=msg2
            )

    def test_invalid_input_model(self):
        r"""Raise `TypeError` when input `model` is invalid."""
        msg1 = 'Must raise `TypeError` when input `model` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_optimizer_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    model=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`model` must be an instance of '
                '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `torch.optim.SGD` or `torch.optim.Adam`."""
        msg = 'Must return `torch.optim.SGD` or `torch.optim.Adam`.'

        test_path = os.path.join(
            self.__class__.test_dir,
            f'optimizer-{self.__class__.checkpoint}.pt'
        )

        for (
                d_emb,
                d_hid,
                dropout,
                model_cstr,
                num_linear_layers,
                num_rnn_layers,
                pad_token_id,
                vocab_size,
                learning_rate,
                (optimizer_class, optimizer_cstr)
        ) in product(
            *self.__class__.model_parameters.values(),
            *self.__class__.optimizer_parameters.values()
        ):
            if vocab_size <= pad_token_id:
                continue

            config = lmp.config.BaseConfig(
                d_emb=d_emb,
                d_hid=d_hid,
                dataset=self.__class__.dataset,
                dropout=dropout,
                experiment=self.__class__.experiment,
                learning_rate=learning_rate,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                optimizer_class=optimizer_class
            )

            model = model_cstr(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )

            optimizer_1 = lmp.util.load_optimizer_by_config(
                checkpoint=-1,
                config=config,
                model=model
            )

            self.assertIsInstance(optimizer_1, optimizer_cstr, msg=msg)

            try:
                # Create test file.
                torch.save(optimizer_1.state_dict(), test_path)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                optimizer_2 = lmp.util.load_optimizer_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config,
                    model=model
                )

                self.assertIsInstance(optimizer_2, optimizer_cstr, msg=msg)
            finally:
                # Clean up test file.
                os.remove(test_path)

    def test_load_result(self):
        r"""Load result must be consistent."""
        msg = 'Inconsistent load result.'

        test_path = os.path.join(
            self.__class__.test_dir,
            f'optimizer-{self.__class__.checkpoint}.pt'
        )

        for (
                d_emb,
                d_hid,
                dropout,
                model_cstr,
                num_linear_layers,
                num_rnn_layers,
                pad_token_id,
                vocab_size,
                learning_rate,
                (optimizer_class, optimizer_cstr)
        ) in product(
            *self.__class__.model_parameters.values(),
            *self.__class__.optimizer_parameters.values()
        ):
            if vocab_size <= pad_token_id:
                continue

            config = lmp.config.BaseConfig(
                d_emb=d_emb,
                d_hid=d_hid,
                dataset=self.__class__.dataset,
                dropout=dropout,
                experiment=self.__class__.experiment,
                learning_rate=learning_rate,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                optimizer_class=optimizer_class
            )

            model = model_cstr(
                d_emb=d_emb,
                d_hid=d_hid,
                dropout=dropout,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )

            try:
                # Create test file.
                ans_optimizer = optimizer_cstr(
                    params=model.parameters(),
                    lr=learning_rate
                )
                torch.save(ans_optimizer.state_dict(), test_path)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                optimizer_1 = lmp.util.load_optimizer_by_config(
                    checkpoint=-1,
                    config=config,
                    model=model
                )
                optimizer_2 = lmp.util.load_optimizer_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config,
                    model=model
                )

                self.assertEqual(
                    len(list(ans_optimizer.state_dict())),
                    len(list(optimizer_1.state_dict())),
                    msg=msg
                )
                self.assertEqual(
                    len(list(ans_optimizer.state_dict())),
                    len(list(optimizer_2.state_dict())),
                    msg=msg
                )

                for p1, p2 in zip(
                        ans_optimizer.state_dict(),
                        optimizer_2.state_dict()
                ):
                    self.assertTrue((p1 == p2), msg=msg)

            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
