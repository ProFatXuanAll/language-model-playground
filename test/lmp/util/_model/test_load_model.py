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
import math
import os
import unittest

from itertools import product
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.model
import lmp.path
import lmp.util


class TestLoadModel(unittest.TestCase):
    r"""Test case for `lmp.util.load_model`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.checkpoint = 10
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.model_parameters = {
            'd_emb': [1, 2],
            'd_hid': [1, 2],
            'dropout': [0.0, 0.1],
            'model': [
                ('rnn', lmp.model.BaseRNNModel),
                ('gru', lmp.model.GRUModel),
                ('lstm', lmp.model.LSTMModel),
                ('res_rnn', lmp.model.BaseResRNNModel),
                ('res_gru', lmp.model.ResGRUModel),
                ('res_lstm', lmp.model.ResLSTMModel),
            ],
            'num_linear_layers': [1, 2],
            'num_rnn_layers': [1, 2],
            'pad_token_id': [0, 1],
            'vocab_size': [5, 10],
        }
        cls.test_dir = os.path.join(lmp.path.DATA_PATH, cls.experiment)
        if os.path.exists(cls.test_dir):
            for model_file in os.listdir(cls.test_dir):
                os.remove(os.path.join(cls.test_dir, model_file))
            os.removedirs(cls.test_dir)
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory and delete dynamic parameters."""
        os.removedirs(cls.test_dir)
        del cls.checkpoint
        del cls.experiment
        del cls.model_parameters
        del cls.test_dir

    def setUp(self):
        r"""Setup fixed parameters."""
        self.checkpoint = -1
        self.d_emb = 1
        self.d_hid = 1
        self.device = torch.device('cpu')
        self.dropout = 0.1
        self.experiment = 'test_util_load_model'
        self.model_class = 'res_rnn'
        self.num_linear_layers = 1
        self.num_rnn_layers = 1
        self.pad_token_id = 0
        self.vocab_size = 5

    def tearDown(self):
        r"""Delete fixed parameters."""
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
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be bigger than or equal to `-1`.',
                    msg=msg2
                )

    def test_invalid_input_d_emb(self):
        r"""Raise exception when input `d_emb` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `d_emb` is '
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
        r"""Raise exception when input `d_hid` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `d_hid` is '
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
        r"""Raise `TypeError` when input `device` is invalid."""
        msg1 = (
            'Must raise `TypeError` when input `device` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
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

            self.assertEqual(
                ctx_man.exception.args[0],
                '`device` must be an instance of `torch.device`.',
                msg=msg2
            )

    def test_invalid_input_dropout(self):
        r"""Raise exception when input `dropout` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `dropout` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, -1.0, 1.1, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
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
        r"""Raise exception when input `experiment` is invalid."""
        msg1 = (
            'Must raise `FileNotFoundError`, `TypeError` or `ValueError` when '
            'input `experiment` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', 'I-DO-NOT-EXIST', b'', (), [], {}, set(),
            object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (FileNotFoundError, TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_model(
                    checkpoint=self.__class__.checkpoint,
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

            if isinstance(ctx_man.exception, FileNotFoundError):
                test_path = os.path.join(
                    lmp.path.DATA_PATH,
                    invalid_input,
                    f'model-{self.__class__.checkpoint}.pt'
                )
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'File {test_path} does not exist.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, TypeError):
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
        r"""Raise exception when input `model_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `model_class` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
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
                    f'model `{invalid_input}` does not support.\nSupported '
                    'options:' +
                    ''.join(list(map(
                        lambda option: f'\n\t--model_class {option}',
                        [
                            'rnn',
                            'gru',
                            'lstm',
                            'res_rnn',
                            'res_gru',
                            'res_lstm',
                        ]
                    ))),
                    msg=msg2
                )

    def test_invalid_input_num_linear_layers(self):
        r"""Raise exception when input `num_linear_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`num_linear_layers` is invalid.'
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
        r"""Raise exception when input `num_rnn_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`num_rnn_layers` is invalid.'
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
        r"""Raise exception when input `pad_token_id` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `pad_token_id` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', (), [], {}, set(), object(), lambda x: x, type, None,
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
        r"""Raise exception when input `vocab_size` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `vocab_size` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j,
            1j, '', b'', (), [], {}, set(), object(), lambda x: x, type, None,
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
                    pad_token_id=self.pad_token_id,
                    vocab_size=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`vocab_size` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`vocab_size` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_pad_token_id_and_vocab_size(self):
        r"""Raise `ValueError` when input `vocab_size <= pad_token_id`."""
        msg1 = (
            'Must raise `ValueError` when input `vocab_size <= pad_token_id`.'
        )
        msg2 = 'Inconsistent error message.'
        examples = ((2, 1), (3, 2), (4, 3), (10, 1))

        for pad_token_id, vocab_size in examples:
            with self.assertRaises(ValueError, msg=msg1) as ctx_man:
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
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`pad_token_id` must be smaller than `vocab_size`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `lmp.model.BaseRNNModel` or `lmp.model.BaseResRNNModel`."""
        msg = (
            'Must return `lmp.model.BaseRNNModel` or '
            '`lmp.model.BaseResRNNModel`.'
        )

        test_path = os.path.join(
            self.__class__.test_dir,
            f'model-{self.__class__.checkpoint}.pt'
        )

        for (
                d_emb,
                d_hid,
                dropout,
                (model_class, model_cstr),
                num_linear_layers,
                num_rnn_layers,
                pad_token_id,
                vocab_size
        ) in product(*self.__class__.model_parameters.values()):
            if vocab_size <= pad_token_id:
                continue

            model_1 = lmp.util.load_model(
                checkpoint=-1,
                d_emb=d_emb,
                d_hid=d_hid,
                device=torch.device('cpu'),
                dropout=dropout,
                experiment=self.__class__.experiment,
                model_class=model_class,
                num_linear_layers=num_linear_layers,
                num_rnn_layers=num_rnn_layers,
                pad_token_id=pad_token_id,
                vocab_size=vocab_size
            )

            self.assertIsInstance(model_1, model_cstr, msg=msg)

            try:
                # Create test file.
                torch.save(model_1.state_dict(), test_path)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                model_2 = lmp.util.load_model(
                    checkpoint=self.__class__.checkpoint,
                    d_emb=d_emb,
                    d_hid=d_hid,
                    device=torch.device('cpu'),
                    dropout=dropout,
                    experiment=self.__class__.experiment,
                    model_class=model_class,
                    num_linear_layers=num_linear_layers,
                    num_rnn_layers=num_rnn_layers,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size
                )

                self.assertIsInstance(model_2, model_cstr, msg=msg)
            finally:
                # Clean up test file.
                os.remove(test_path)

    def test_load_result(self):
        r"""Load result must be consistent."""
        msg = 'Inconsistent load result.'

        test_path = os.path.join(
            self.__class__.test_dir,
            f'model-{self.__class__.checkpoint}.pt'
        )

        for (
                d_emb,
                d_hid,
                dropout,
                (model_class, model_cstr),
                num_linear_layers,
                num_rnn_layers,
                pad_token_id,
                vocab_size
        ) in product(*self.__class__.model_parameters.values()):
            if vocab_size <= pad_token_id:
                continue
            try:
                # Create test file.
                ans_model = model_cstr(
                    d_emb=d_emb,
                    d_hid=d_hid,
                    dropout=dropout,
                    num_linear_layers=num_linear_layers,
                    num_rnn_layers=num_rnn_layers,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size
                )
                torch.save(ans_model.state_dict(), test_path)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                model_1 = lmp.util.load_model(
                    checkpoint=-1,
                    d_emb=d_emb,
                    d_hid=d_hid,
                    device=torch.device('cpu'),
                    dropout=dropout,
                    experiment=self.__class__.experiment,
                    model_class=model_class,
                    num_linear_layers=num_linear_layers,
                    num_rnn_layers=num_rnn_layers,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size
                )

                model_2 = lmp.util.load_model(
                    checkpoint=self.__class__.checkpoint,
                    d_emb=d_emb,
                    d_hid=d_hid,
                    device=torch.device('cpu'),
                    dropout=dropout,
                    experiment=self.__class__.experiment,
                    model_class=model_class,
                    num_linear_layers=num_linear_layers,
                    num_rnn_layers=num_rnn_layers,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size
                )

                self.assertEqual(
                    len(list(ans_model.parameters())),
                    len(list(model_1.parameters())),
                    msg=msg
                )
                self.assertEqual(
                    len(list(ans_model.parameters())),
                    len(list(model_2.parameters())),
                    msg=msg
                )

                for p1, p2 in zip(ans_model.parameters(),
                                  model_2.parameters()):
                    self.assertTrue((p1 == p2).all().item(), msg=msg)

            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
