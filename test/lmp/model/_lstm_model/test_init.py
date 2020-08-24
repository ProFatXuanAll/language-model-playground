r"""Test `lmp.model.LSTMModel.__init__`.

Usage:
    python -m unittest test.lmp.model._lstm_model.test_init
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

from lmp.model import BaseRNNModel
from lmp.model import LSTMModel


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.model.LSTMModel.__init__`."""

    @classmethod
    def setUpClass(cls):
        cls.batch_range = [1, 2]
        cls.d_emb_range = [1, 10]
        cls.d_hid_range = [1, 10]
        cls.dropout_range = [0.0, 0.1, 0.5, 1.0]
        cls.num_linear_layers_range = [1, 2]
        cls.num_rnn_layers_range = [1, 2]
        cls.pad_token_id_range = [0, 1, 2, 3]
        cls.vocab_size_range = [1, 5]
        cls.sequence_range = list(range(1, 5))

    @classmethod
    def tearDownClass(cls):
        del cls.batch_range
        del cls.d_emb_range
        del cls.d_hid_range
        del cls.dropout_range
        del cls.num_linear_layers_range
        del cls.num_rnn_layers_range
        del cls.pad_token_id_range
        del cls.sequence_range
        del cls.vocab_size_range
        gc.collect()

    def setUp(self):
        r"""Setup hyperparameters and construct `LSTMModel`."""
        self.model_objs = []
        cls = self.__class__
        for d_emb in cls.d_emb_range:
            for d_hid in cls.d_hid_range:
                for dropout in cls.dropout_range:
                    for num_linear_layers in cls.num_linear_layers_range:
                        for num_rnn_layers in cls.num_rnn_layers_range:
                            for pad_token_id in cls.pad_token_id_range:
                                for vocab_size in cls.vocab_size_range:
                                    # skip invalid construct.
                                    if vocab_size <= pad_token_id:
                                        continue

                                    model = LSTMModel(
                                        d_emb=d_emb,
                                        d_hid=d_hid,
                                        dropout=dropout,
                                        num_linear_layers=num_linear_layers,
                                        num_rnn_layers=num_rnn_layers,
                                        pad_token_id=pad_token_id,
                                        vocab_size=vocab_size
                                    )
                                    self.model_objs.append({
                                        'd_emb': d_emb,
                                        'd_hid': d_hid,
                                        'dropout': dropout,
                                        'model': model,
                                        'num_linear_layers': num_linear_layers,
                                        'num_rnn_layers': num_rnn_layers,
                                        'pad_token_id': pad_token_id,
                                        'vocab_size': vocab_size,
                                    })

    def tearDown(self):
        r"""Delete model instances."""
        del self.model_objs
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistenct method signature.'

        self.assertEqual(
            inspect.signature(LSTMModel.__init__),
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
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
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
                LSTMModel(
                    d_emb=invalid_input,
                    d_hid=1,
                    dropout=0.1,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=1
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
                LSTMModel(
                    d_emb=1,
                    d_hid=invalid_input,
                    dropout=0.1,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=1
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
                LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=invalid_input,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=1
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
                LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_linear_layers=invalid_input,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=1
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
                LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_linear_layers=1,
                    num_rnn_layers=invalid_input,
                    pad_token_id=0,
                    vocab_size=1
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
                LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=invalid_input,
                    vocab_size=1
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
                LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
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
                LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.1,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size,
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`pad_token_id` must be smaller than `vocab_size`.',
                msg=msg2
            )

    def test_inherit(self):
        r""""Is subclass of `lmp.model.BaseRNNModel`."""
        msg = 'Must be subclass of `lmp.model.BaseRNNModel`.'

        for model_obj in self.model_objs:
            self.assertIsInstance(model_obj['model'], BaseRNNModel, msg=msg)

    def test_instance_attributes(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        examples = (
            ('emb_layer', torch.nn.Embedding),
            ('emb_dropout', torch.nn.Dropout),
            ('proj_emb_to_hid', torch.nn.Sequential),
            ('rnn_layer', torch.nn.LSTM),
            ('proj_hid_to_emb', torch.nn.Sequential),
        )

        for attr, attr_type in examples:
            for model_obj in self.model_objs:
                model = model_obj['model']
                self.assertTrue(hasattr(model, attr), msg=msg1.format(attr))
                self.assertIsInstance(
                    getattr(model, attr),
                    attr_type,
                    msg=msg2.format(attr, attr_type.__name__)
                )

    def test_emb_layer(self):
        r"""Declare correct embedding layer with dropout."""
        msg = 'Must declare correct embedding layer with dropout.'
        examples = (
            (
                model_obj['model'].emb_layer,
                model_obj['model'].emb_dropout,
                model_obj['d_emb'],
                model_obj['dropout'],
                model_obj['pad_token_id'],
                model_obj['vocab_size']
            )
            for model_obj in self.model_objs
        )

        for (
                emb_layer,
                emb_dropout,
                d_emb,
                dropout,
                pad_token_id,
                vocab_size
        ) in examples:
            self.assertIsInstance(emb_layer, torch.nn.Embedding, msg=msg)
            self.assertEqual(emb_layer.num_embeddings, vocab_size, msg=msg)
            self.assertEqual(emb_layer.embedding_dim, d_emb, msg=msg)
            self.assertEqual(emb_layer.padding_idx, pad_token_id, msg=msg)
            self.assertIsInstance(emb_dropout, torch.nn.Dropout, msg=msg)
            self.assertEqual(emb_dropout.p, dropout, msg=msg)

    def test_proj_emb_to_hid(self):
        r"""Declare correct projection layer with dropout."""
        msg = 'Must declare correct projection layer with dropout.'
        examples = (
            (
                model_obj['model'].proj_emb_to_hid,
                model_obj['d_emb'],
                model_obj['d_hid'],
                model_obj['dropout'],
            )
            for model_obj in self.model_objs
        )

        for proj_layer, d_emb, d_hid, dropout in examples:
            self.assertEqual(len(proj_layer), 3, msg=msg)
            self.assertIsInstance(proj_layer[0], torch.nn.Linear, msg=msg)
            self.assertEqual(proj_layer[0].in_features, d_emb, msg=msg)
            self.assertEqual(proj_layer[0].out_features, d_hid, msg=msg)
            self.assertIsInstance(proj_layer[1], torch.nn.ReLU, msg=msg)
            self.assertIsInstance(proj_layer[2], torch.nn.Dropout, msg=msg)
            self.assertEqual(proj_layer[2].p, dropout, msg=msg)

    def test_lstm_layer(self):
        r"""Declare correct LSTM layer(s)."""
        msg = 'Must declare correct LSTM layer(s).'
        examples = (
            (
                model_obj['model'].rnn_layer,
                model_obj['d_hid'],
                model_obj['dropout'],
                model_obj['num_rnn_layers'],
            )
            for model_obj in self.model_objs
        )

        for rnn_layer, d_hid, dropout, num_rnn_layers in examples:
            self.assertIsInstance(rnn_layer, torch.nn.LSTM, msg=msg)
            self.assertEqual(rnn_layer.input_size, d_hid, msg=msg)
            self.assertEqual(rnn_layer.hidden_size, d_hid, msg=msg)
            self.assertEqual(rnn_layer.num_layers, num_rnn_layers, msg=msg)
            self.assertTrue(rnn_layer.batch_first, msg=msg)

            if num_rnn_layers == 1:
                self.assertEqual(rnn_layer.dropout, 0.0, msg=msg)
            else:
                self.assertEqual(rnn_layer.dropout, dropout, msg=msg)

    def test_proj_hid_to_emb(self):
        r"""Declare correct projection layer with dropout."""
        msg = 'Must declare correct projection layer with dropout.'
        examples = (
            (
                model_obj['model'].proj_hid_to_emb,
                model_obj['d_emb'],
                model_obj['d_hid'],
                model_obj['dropout'],
                model_obj['num_linear_layers'],
            )
            for model_obj in self.model_objs
        )

        for proj_layer, d_emb, d_hid, dropout, num_linear_layers in examples:
            self.assertEqual(
                len(proj_layer),
                3 * num_linear_layers + 1,
                msg=msg
            )
            for i in range(0, num_linear_layers - 1, 3):
                self.assertIsInstance(proj_layer[i], torch.nn.Dropout, msg=msg)
                self.assertEqual(proj_layer[i].p, dropout, msg=msg)
                self.assertIsInstance(
                    proj_layer[i + 1],
                    torch.nn.Linear,
                    msg=msg
                )
                self.assertEqual(proj_layer[i + 1].in_features, d_hid, msg=msg)
                self.assertEqual(
                    proj_layer[i + 1].out_features,
                    d_hid,
                    msg=msg
                )
                self.assertIsInstance(
                    proj_layer[i + 2],
                    torch.nn.ReLU,
                    msg=msg
                )
            self.assertIsInstance(proj_layer[-4], torch.nn.Dropout, msg=msg)
            self.assertEqual(proj_layer[-4].p, dropout, msg=msg)
            self.assertIsInstance(proj_layer[-3], torch.nn.Linear, msg=msg)
            self.assertEqual(proj_layer[-3].in_features, d_hid, msg=msg)
            self.assertEqual(proj_layer[-3].out_features, d_emb, msg=msg)
            self.assertIsInstance(proj_layer[-2], torch.nn.ReLU, msg=msg)
            self.assertIsInstance(proj_layer[-1], torch.nn.Dropout, msg=msg)
            self.assertEqual(proj_layer[-1].p, dropout, msg=msg)


if __name__ == '__main__':
    unittest.main()
