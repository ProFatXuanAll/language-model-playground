r"""Test `lmp.util.load_model.`.

Usage:
    python -m unittest test.lmp.util._model.test_load_model
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import json
import math
import os
import unittest

from itertools import product
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestLoadModel(unittest.TestCase):
    r"""Test Case for `lmp.util.load_model`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory."""
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.test_dir = os.path.join(
            lmp.path.DATA_PATH,
            cls.experiment
        )
        os.makedirs(cls.test_dir)

        cls.checkpoint = 1226
        cls.parameters = {
            'd_emb': [5, 6],
            'd_hid': [7, 9],
            'dropout': [0.1, 0.5],
            'model_class': [
                'rnn',
                'gru',
                'lstm',
                'res_rnn',
                'res_gru',
                'res_lstm'
            ],
            'num_linear_layers': [3, 6],
            'num_rnn_layers': [2, 5],
            'pad_token_id': [0, 1, 2, 3],
            'vocab_size': [10, 15]
        }
        cls.param_values = [v for v in cls.parameters.values()]


    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory."""
        os.removedirs(cls.test_dir)
        del cls.checkpoint
        del cls.experiment
        del cls.parameters
        del cls.param_values
        del cls.test_dir

    def setUp(self):
        r"""Set up parameters for `load_model`."""
        self.checkpoint = -1
        self.d_emb = 1
        self.d_hid = 1
        self.device = torch.tensor([10]).device
        self.dropout = 0.1
        self.experiment = 'test_util_load_model'
        self.model_class = 'res_rnn'
        self.num_linear_layers = 1
        self.num_rnn_layers = 1
        self.pad_token_id = 0
        self.vocab_size = 10

    def tearDown(self):
        r"""Delete parameters for `load_model`."""
        del self.checkpoint
        del self.d_emb
        del self.d_hid
        del self.device
        del self.dropout
        del self.experiment
        del self.model_class
        del self.num_linear_layers
        del self.num_rnn_layers
        del self.pad_token_id
        del self.vocab_size
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_model),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='d_emb',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='d_hid',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='device',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.device,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='dropout',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='model_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='num_linear_layers',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='num_rnn_layers',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='pad_token_id',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='vocab_size',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=Union[
                    lmp.model.BaseRNNModel,
                    lmp.model.BaseResRNNModel
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
                lmp.util.load_model(
                    checkpoint=invalid_input,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
                )

    def test_invalid_input_d_emb(self):
        r"""Raise when `d_emb` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_emb` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=invalid_input,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_d_hid(self):
        r"""Raise when `d_hid` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_hid` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=invalid_input,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_device(self):
        r"""Raise when `device` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `device` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=invalid_input,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`device` must be an instance of `torch.device`.',
                    msg=msg2
                )

    def test_invalid_input_dropout(self):
        r"""Raise when `dropout` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `dropout` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, -1.0, 1.1, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=invalid_input,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must be an instance of `float`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must range from `0.0` to `1.0`.',
                    msg=msg2
                )

    def test_invalid_input_experiment(self):
        r"""Raise when `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `experiment` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=invalid_input,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must be an instance of `str`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must not be empty.',
                    msg=msg2
                )

    def test_invalid_input_model_class(self):
        r"""Raise when `model_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `model_class` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=invalid_input,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model_class` must be an instance of `str`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model_class` must not be empty.',
                    msg=msg2
                )

    def test_invalid_input_num_linear_layers(self):
        r"""Raise when `num_linear_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `num_linear_layers` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=invalid_input,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_linear_layers` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_linear_layers` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_num_rnn_layers(self):
        r"""Raise when `num_rnn_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `num_rnn_layers` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=invalid_input,
                    pad_token_id=self.pad_token_id,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_rnn_layers` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_rnn_layers` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_pad_token_id(self):
        r"""Raise when `pad_token_id` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `pad_token_id` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                (TypeError, ValueError),
                msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=invalid_input,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`pad_token_id` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`pad_token_id` must be bigger than or equal to `0`.',
                    msg=msg2
                )

    def test_invalid_input_vocab_size(self):
        r"""Raise when `vocab_size` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `vocab_size` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.checkpoint,
                    d_emb=self.d_emb,
                    d_hid=self.d_hid,
                    device=self.device,
                    dropout=self.dropout,
                    experiment=self.experiment,
                    model_class=self.model_class,
                    num_linear_layers=self.num_linear_layers,
                    num_rnn_layers=self.num_rnn_layers,
                    pad_token_id=self.pad_token_id,
                    vocab_size=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`vocab_size` must be an instance of `int`.',
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `lmp.model.BaseRNNModel` or `lmp.model.BaseResRNNModel`."""
        msg = (
            'Must return `lmp.model.BaseRNNModel` or '
            '`lmp.model.BaseResRNNModel`.'
        )
        device = torch.tensor([10]).device
        for(
            d_emb,
            d_hid,
            dropout,
            model_class,
            num_linear_layers,
            num_rnn_layers,
            pad_token_id,
            vocab_size
        ) in product(*self.__class__.param_values):
            if vocab_size <= pad_token_id:
                continue
            model = lmp.util.load_model(
                checkpoint=-1,
                d_emb=d_emb,
                d_hid=d_hid,
                device=device,
                dropout=dropout,
                experiment='util_load_model_unittest',
                model_class=model_class,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )
            try:
                self.assertIsInstance(model, lmp.model.BaseRNNModel, msg=msg)
            except AssertionError:
                self.assertIsInstance(model, lmp.model.BaseResRNNModel, msg=msg)

    def test_load_model_by_checkpoint(self):
        r"""Save result must be consistent."""
        msg1 = 'Must save as `model.json`.'
        msg2 = 'Inconsistent save result.'

        examples = (
            {
                'd_emb': 10,
                'd_hid': 10,
                'dropout': 0.1,
                'num_rnn_layers': 1,
                'num_linear_layers': 2,
                'pad_token_id': 0,
                'vocab_size': 10
            },
            {
                'd_emb': 5,
                'd_hid': 6,
                'dropout': 0.25,
                'num_rnn_layers': 3,
                'num_linear_layers': 1,
                'pad_token_id': 0,
                'vocab_size': 40
            },
        )

        attr_examples = (
            (
                'emb_layer',
                {
                    'num_embeddings': 'vocab_size',
                    'embedding_dim': 'd_emb'
                },
            ),
            (
                'emb_dropout',
                {
                    'p': 'dropout'
                }
            ),
        )

        proj_emb_to_hid_examples = (
            (
                'in_features',
                'd_emb',
            ),
            (
                'out_features', 
                'd_hid',
            ),
        )

        for ans_attributes in examples:
            test_path = os.path.join(
                self.__class__.test_dir,
                f'model-{self.__class__.checkpoint}.pt'
            )
            try:
                # Create test file.
                model_test = lmp.model.BaseResRNNModel(**ans_attributes)
                torch.save(
                    model_test.state_dict(),
                    test_path
                )
                self.assertTrue(os.path.exists(test_path), msg=msg1)

                model = lmp.util.load_model(
                    checkpoint=self.__class__.checkpoint,
                    d_emb=ans_attributes['d_emb'],
                    d_hid=ans_attributes['d_hid'],
                    device=torch.tensor([10]).device,
                    dropout=ans_attributes['dropout'],
                    experiment=self.__class__.experiment,
                    model_class='res_rnn',
                    num_linear_layers=ans_attributes['num_linear_layers'],
                    num_rnn_layers=ans_attributes['num_rnn_layers'],
                    pad_token_id=ans_attributes['pad_token_id'],
                    vocab_size=ans_attributes['vocab_size']
                )
                for attr_key, attr_dict in attr_examples:
                    self.assertTrue(hasattr(model, attr_key), msg=msg2)
                    model_attr = getattr(model, attr_key)
                        
                    for model_key, ans_key in attr_dict.items():
                        self.assertEqual(
                            getattr(model_attr, model_key),
                            ans_attributes[ans_key],
                            msg=msg2
                        )
                self.assertEqual(
                    len(model.rnn_layer),
                    ans_attributes['num_rnn_layers'],
                    msg=msg2
                )
                for layer in model.proj_emb_to_hid:
                    if isinstance(layer, torch.nn.Dropout):
                        continue
                    for layer_key, ans_key in proj_emb_to_hid_examples:
                        self.assertEqual(
                            getattr(layer, layer_key),
                            ans_attributes[ans_key],
                            msg=msg2
                        )
                nums = 0
                for layer in model.proj_hid_to_emb:
                    if isinstance(layer, torch.nn.Linear):
                        nums += 1
                self.assertEqual(
                    nums-1,
                    ans_attributes['num_linear_layers'],
                    msg=msg2
                )
            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
