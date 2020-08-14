r"""Test `lmp.model.ResLSTMModel.__init__`.

Usage:
    python -m unittest test/lmp/model/_res_lstm_model/test_init.py
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

# 3rd-party modules

import torch
import torch.nn

# self-made modules

import lmp.model

from lmp.model import ResLSTMModel


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.model.ResLSTMModel.__init__`."""

    def setUp(self):
        r"""Set up hyper parameters and construct ResLSTMModel"""
        self.d_emb = 10
        self.d_hid = 10
        self.dropout = 0.1
        self.num_rnn_layers = 1
        self.num_linear_layers = 1
        self.pad_token_id = 0
        self.vocab_size = 30

        self.model_parameters = (
            (
                ('d_emb', self.d_emb),
                ('d_hid', self.d_hid),
                ('dropout', self.dropout),
                ('num_rnn_layers', self.num_rnn_layers),
                ('num_linear_layers', self.num_linear_layers),
                ('pad_token_id', self.pad_token_id),
                ('vocab_size', self.vocab_size),
            ),
        )

    def tearDown(self):
        r"""Delete parameters and models."""
        del self.d_emb
        del self.d_hid
        del self.dropout
        del self.num_rnn_layers
        del self.num_linear_layers
        del self.pad_token_id
        del self.vocab_size
        del self.model_parameters
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(ResLSTMModel.__init__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
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
                        name='dropout',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='num_rnn_layers',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='num_linear_layers',
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
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input_d_emb(self):
        r"""Raise when `d_emb` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_emb` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                ResLSTMModel(
                    d_emb=invalid_input,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_emb` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_d_hid(self):
        r"""Raise when `d_hid` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `d_hid` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                ResLSTMModel(
                    d_emb=1,
                    d_hid=invalid_input,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`d_hid` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_dropout(self):
        r"""Raise when `dropout` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `dropout` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, -1.0, 1.1, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=invalid_input,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )
            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must be an instance of `float`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dropout` must range from `0.0` to `1.0`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_num_linear_layers(self):
        r"""Raise when `num_linear_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `num_linear_layers` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=invalid_input,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_linear_layers` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_linear_layers` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_num_rnn_layers(self):
        r"""Raise when `num_rnn_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `num_rnn_layers` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=invalid_input,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_rnn_layers` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`num_rnn_layers` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_pad_token_id(self):
        r"""Raise when `pad_token_id` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `pad_token_id` '
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
                ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=invalid_input,
                    vocab_size=10
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`pad_token_id` must be an instance of `int`.',
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
                ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_rnn_layers=1,
                    num_linear_layers=1,
                    pad_token_id=0,
                    vocab_size=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`vocab_size` must be an instance of `int`.',
                    msg=msg2
                )

    def test_instance_attribute_embedding_layer(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.randint(low=0, high=9, size=(5, 10)),
            torch.randint(low=0, high=9, size=(10, 15)),
        )

        emb_layer = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_emb,
            padding_idx=self.pad_token_id
        )

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                ResLSTMModel(*pos),
                ResLSTMModel(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'emb_layer'),
                msg=msg1.format('emb_layer')
            )
            self.assertIsInstance(
                model.emb_layer,
                type(emb_layer),
                msg=msg2.format(
                    'emb_layer',
                    type(emb_layer).__name__
                )
            )
            for x in examples:
                ht = model.emb_layer(x)
                ans_out = emb_layer(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )

    def test_instance_attribute_emb_dropout(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_emb),
            torch.rand(10, 20, self.d_emb),
        )

        emb_dropout = torch.nn.Dropout(self.dropout)

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                ResLSTMModel(*pos),
                ResLSTMModel(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'emb_dropout'),
                msg=msg1.format('emb_dropout')
            )
            self.assertIsInstance(
                model.emb_dropout,
                type(emb_dropout),
                msg=msg2.format('emb_dropout', type(emb_dropout).__name__)
            )
            for x in examples:
                ht = model.emb_dropout(x)
                ans_out = emb_dropout(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )

    def test_instance_attribute_proj_emb_to_hid(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_emb),
            torch.rand(10, 20, self.d_emb),
        )

        proj_emb_to_hid = []
        proj_emb_to_hid.append(
            torch.nn.Linear(
                in_features=self.d_emb,
                out_features=self.d_hid
            )
        )
        proj_emb_to_hid.append(torch.nn.Dropout(self.dropout))
        proj_emb_to_hid = torch.nn.Sequential(*proj_emb_to_hid)

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                ResLSTMModel(*pos),
                ResLSTMModel(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'proj_emb_to_hid'),
                msg=msg1.format('proj_emb_to_hid')
            )
            self.assertIsInstance(
                model.proj_emb_to_hid,
                type(proj_emb_to_hid),
                msg=msg2.format(
                    'proj_emb_to_hid',
                    type(proj_emb_to_hid).__name__
                )
            )
            for x in examples:
                ht = model.proj_emb_to_hid(x)
                ans_out = proj_emb_to_hid(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )

    def test_instance_attribute_rnn_layer(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_hid),
            torch.rand(10, 20, self.d_hid),
        )

        res_rnn_block = lmp.model.ResLSTMBlock(
            d_hid=self.d_hid,
            dropout=self.dropout
        )
        dropout = torch.nn.Dropout(self.dropout)

        rnn_blocks = []
        for _ in range(self.num_rnn_layers):
            rnn_blocks.append(res_rnn_block)
        rnn_blocks.append(dropout)
        rnn_layer = torch.nn.Sequential(*rnn_blocks)

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                ResLSTMModel(*pos),
                ResLSTMModel(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'rnn_layer'),
                msg=msg1.format('rnn_layer')
            )
            self.assertIsInstance(
                model.rnn_layer,
                type(rnn_layer),
                msg=msg2.format('rnn_layer', type(rnn_layer).__name__)
            )
            for x in examples:
                ht = model.rnn_layer(x)
                ans_out = rnn_layer(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )

    def test_instance_attribute_proj_hid_to_emb(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be instance of `{}`.'
        msg3 = 'Return size must be {}.'
        examples = (
            torch.rand(5, 10, self.d_hid),
            torch.rand(10, 20, self.d_hid),
        )

        linear = torch.nn.Linear(
            in_features=self.d_hid,
            out_features=self.d_hid
        )
        act_fn = torch.nn.ReLU()
        dropout = torch.nn.Dropout(self.dropout)
        output_linear = torch.nn.Linear(
            in_features=self.d_hid,
            out_features=self.d_emb
        )
        proj_hid_to_emb = []
        for _ in range(self.num_linear_layers):
            proj_hid_to_emb.append(linear)
            proj_hid_to_emb.append(act_fn)
            proj_hid_to_emb.append(dropout)

        proj_hid_to_emb.append(output_linear)
        proj_hid_to_emb = torch.nn.Sequential(*proj_hid_to_emb)

        for parameters in self.model_parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            models = [
                ResLSTMModel(*pos),
                ResLSTMModel(**kwargs),
            ]

        for model in models:
            self.assertTrue(
                hasattr(model, 'proj_hid_to_emb'),
                msg=msg1.format('proj_hid_to_emb')
            )
            self.assertIsInstance(
                model.proj_hid_to_emb,
                type(proj_hid_to_emb),
                msg=msg2.format(
                    'proj_hid_to_emb',
                    type(proj_hid_to_emb).__name__
                )
            )
            for x in examples:
                ht = model.proj_hid_to_emb(x)
                ans_out = proj_hid_to_emb(x)
                self.assertEqual(
                    ht.size(),
                    ans_out.size(),
                    msg=msg3.format(ans_out.size())
                )


if __name__ == '__main__':
    unittest.main()
