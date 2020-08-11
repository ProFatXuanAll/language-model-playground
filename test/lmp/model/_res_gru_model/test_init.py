r"""Test `lmp.model.ResGRUModel.__init__`.

Usage:
    python -m unittest test/lmp/model/_res_gru_model/test_init.py
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

from lmp.model import ResGRUModel


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.model.ResGRUModel.__init__`."""

    def setUp(self):
        r"""Set up hyper parameters and construct ResGRUModel"""
        self.d_emb = 1
        self.d_hid = 1
        self.dropout = 0.1
        self.num_rnn_layers = 1
        self.num_linear_layers = 1
        self.pad_token_id = 0
        self.vocab_size = 1

        Parameters = (
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

        for parameters in Parameters:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            self.models = [
                ResGRUModel(*pos),
                ResGRUModel(**kwargs),
            ]

    def tearDown(self):
        r"""Delete parameters and models."""
        del self.d_emb
        del self.d_hid
        del self.dropout
        del self.num_rnn_layers
        del self.num_linear_layers
        del self.pad_token_id
        del self.vocab_size
        del self.models
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(ResGRUModel.__init__),
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
                ResGRUModel(
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
                ResGRUModel(
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
                ResGRUModel(
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
                ResGRUModel(
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
                ResGRUModel(
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
                ResGRUModel(
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
                ResGRUModel(
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
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        msg3 = 'Instance attribute `{}` must be `{}`.'

        embedding_layer = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_emb,
            padding_idx=self.pad_token_id
        )

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'embedding_layer'),
                msg=msg1.format('embedding_layer')
            )
            self.assertIsInstance(
                getattr(model, 'embedding_layer'),
                type(embedding_layer),
                msg=msg2.format(
                    'embedding_layer',
                    type(embedding_layer).__name__
                )
            )
            self.assertEqual(
                getattr(model, 'embedding_layer').weight,
                embedding_layer.weight,
                msg=msg3.format('embedding_layer', embedding_layer)
            )

    def test_instance_attribute_dropout(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        msg3 = 'Instance {} attribute `{}` must need `{}`.'

        dropout_examples = (
            ('p', self.dropout),
        )

        dropout = torch.nn.Dropout(self.dropout)

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'dropout'),
                msg=msg1.format('dropout')
            )
            self.assertIsInstance(
                getattr(model, 'dropout'),
                type(dropout),
                msg=msg2.format('dropout', type(dropout).__name__)
            )
            dropout_layer = getattr(model, 'dropout')
            for dropout_attr, dropout_attr_val in dropout_examples:
                self.assertEqual(
                    getattr(dropout_layer, dropout_attr),
                    getattr(dropout, dropout_attr),
                    msg=msg3.format('dropout', dropout_attr, dropout_attr_val)
                )

    def test_instance_attribute_proj_emb_to_hid(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        msg3 = 'Instance {} attribute `{}` must include `{}`.'
        msg4 = 'Inconsitent activation function.'

        linear_examples = (
            ('in_features', self.d_emb),
            ('out_features', self.d_hid),
        )
        dropout_examples = (
            ('p', self.dropout),
        )

        linear = torch.nn.Linear(
            in_features=self.d_emb,
            out_features=self.d_hid
        )
        dropout = torch.nn.Dropout(self.dropout)

        proj_emb_to_hid = []
        proj_emb_to_hid.append(
            torch.nn.Linear(
                in_features=self.d_emb,
                out_features=self.d_hid
            )
        )
        proj_emb_to_hid.append(torch.nn.Dropout(self.dropout))
        proj_emb_to_hid = torch.nn.Sequential(*proj_emb_to_hid)

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'proj_emb_to_hid'),
                msg=msg1.format('proj_emb_to_hid')
            )
            self.assertIsInstance(
                getattr(model, 'proj_emb_to_hid'),
                type(proj_emb_to_hid),
                msg=msg2.format(
                    'proj_emb_to_hid',
                    type(proj_emb_to_hid).__name__
                )
            )
            model_layer = getattr(model, 'proj_emb_to_hid')
            for layer in model_layer:
                if isinstance(layer, torch.nn.modules.dropout.Dropout):
                    for dropout_attr, dropout_attr_val in dropout_examples:
                        self.assertEqual(
                            getattr(
                                layer, dropout_attr), getattr(
                                dropout, dropout_attr), msg=msg3.format(
                                'dropout', dropout_attr, dropout_attr_val))
                    continue
                for linear_attr, linear_attr_val in linear_examples:
                    self.assertEqual(
                        getattr(layer, linear_attr),
                        getattr(linear, linear_attr),
                        msg=msg3.format('linear', linear_attr, linear_attr_val)
                    )

    def test_instance_attribute_rnn_layer(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        msg3 = 'Instance {} attribute `{}` must include `{}`.'

        rnn_examples = (
            (
                'rnn_layer',
                torch.nn.GRU(
                    input_size=self.d_hid,
                    hidden_size=self.d_hid,
                    batch_first=True
                )
            ),
            ('dropout', torch.nn.Dropout(self.dropout)),
            ('act_fn', torch.nn.ReLU())
        )
        dropout_examples = (
            ('p', self.dropout),
        )
        res_rnn_block = lmp.model.ResGRUBlock(
            d_hid=self.d_hid,
            dropout=self.dropout
        )
        dropout = torch.nn.Dropout(self.dropout)

        rnn_blocks = []
        for _ in range(self.num_rnn_layers):
            rnn_blocks.append(res_rnn_block)
        rnn_blocks.append(dropout)
        rnn_layer = torch.nn.Sequential(*rnn_blocks)

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'rnn_layer'),
                msg=msg1.format('rnn_layer')
            )
            self.assertIsInstance(
                getattr(model, 'rnn_layer'),
                type(rnn_layer),
                msg=msg2.format('rnn_layer', type(rnn_layer).__name__)
            )
            model_layer = getattr(model, 'rnn_layer')
            for layer in model_layer:
                if isinstance(layer, torch.nn.modules.dropout.Dropout):
                    for dropout_attr, dropout_attr_val in dropout_examples:
                        self.assertEqual(
                            getattr(
                                layer, dropout_attr), getattr(
                                dropout, dropout_attr), msg=msg3.format(
                                'dropout', dropout_attr, dropout_attr_val))
                    continue
                for rnn_attr, rnn_attr_val in rnn_examples:
                    self.assertIsInstance(
                        getattr(layer, rnn_attr),
                        type(rnn_attr_val),
                        msg=msg3.format('rnn', rnn_attr, rnn_attr_val)
                    )

    def test_instance_attribute_proj_hid_to_emb(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        msg3 = 'Instance {} attribute `{}` must include `{}`.'
        msg4 = 'Inconsitent activation function.'

        linear_examples = (
            ('in_features', self.d_hid),
            ('out_features', self.d_hid),
        )
        dropout_examples = (
            ('p', self.dropout),
        )
        output_linear_examples = (
            ('in_features', self.d_hid),
            ('out_features', self.d_emb),
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

        for model in self.models:
            self.assertTrue(
                hasattr(model, 'proj_hid_to_emb'),
                msg=msg1.format('proj_hid_to_emb')
            )
            self.assertIsInstance(
                getattr(model, 'proj_hid_to_emb'),
                type(proj_hid_to_emb),
                msg=msg2.format(
                    'proj_hid_to_emb',
                    type(proj_hid_to_emb).__name__
                )
            )
            model_layer = getattr(model, 'proj_hid_to_emb')
            for layer in model_layer:
                if isinstance(layer, torch.nn.modules.dropout.Dropout):
                    for dropout_attr, dropout_attr_val in dropout_examples:
                        self.assertEqual(
                            getattr(
                                layer, dropout_attr), getattr(
                                dropout, dropout_attr), msg=msg3.format(
                                'dropout', dropout_attr, dropout_attr_val))
                    continue
                if isinstance(layer, torch.nn.modules.activation.ReLU):
                    self.assertEqual(
                        layer(torch.tensor([0])),
                        act_fn(torch.tensor([0])),
                        msg=msg4
                    )
                    continue
                if layer == model_layer[-1]:
                    for linear_attr, linear_attr_val in output_linear_examples:
                        self.assertEqual(
                            getattr(
                                layer, linear_attr), getattr(
                                linear, linear_attr), msg=msg3.format(
                                'linear', linear_attr, linear_attr_val))
                else:
                    for linear_attr, linear_attr_val in linear_examples:
                        self.assertEqual(
                            getattr(
                                layer, linear_attr), getattr(
                                output_linear, linear_attr), msg=msg3.format(
                                'linear', linear_attr, linear_attr_val))


if __name__ == '__main__':
    unittest.main()
