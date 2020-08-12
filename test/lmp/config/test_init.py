r"""Test `lmp.config.BaseConfig.__init__`.

Usage:
    python -m unittest test/lmp/config/test_init.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

# self-made modules

from lmp.config import BaseConfig


class TestInit(unittest.TestCase):
    r"""Test case for `lmp.config.BaseConfig.__init__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseConfig.__init__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='batch_size',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='checkpoint_step',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=500
                    ),
                    inspect.Parameter(
                        name='d_emb',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='d_hid',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=''
                    ),
                    inspect.Parameter(
                        name='dropout',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=0.1
                    ),
                    inspect.Parameter(
                        name='epoch',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=10
                    ),
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=''
                    ),
                    inspect.Parameter(
                        name='is_uncased',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=False
                    ),
                    inspect.Parameter(
                        name='learning_rate',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=1e-4
                    ),
                    inspect.Parameter(
                        name='max_norm',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
                        default=1.0
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=60
                    ),
                    inspect.Parameter(
                        name='min_count',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='model_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default='lstm'
                    ),
                    inspect.Parameter(
                        name='num_linear_layers',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='num_rnn_layers',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='optimizer_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default='adam'
                    ),
                    inspect.Parameter(
                        name='seed',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=1
                    ),
                    inspect.Parameter(
                        name='tokenizer_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default='char_dict'
                    )
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input_batch_size(self):
        r"""Raise when input `batch_size` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `batch_size` '
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
                BaseConfig(batch_size=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`batch_size` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`batch_size` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_checkpoint_step(self):
        r"""Raise when input `checkpoint_step` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`checkpoint_step` is invalid.'
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
                BaseConfig(checkpoint_step=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint_step` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint_step` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_d_emb(self):
        r"""Raise when input `d_emb` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `d_emb` is '
            'invalid.'
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
                BaseConfig(d_emb=invalid_input)

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
        r"""Raise when input `d_hid` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `d_hid` is '
            'invalid.'
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
                BaseConfig(d_hid=invalid_input)

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

    def test_invalid_input_dataset(self):
        r"""Raise when input `dataset` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `dataset` is '
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
                BaseConfig(dataset=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must not be empty.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_dropout(self):
        r"""Raise when input `dropout` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `dropout` is '
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
                BaseConfig(dataset='test', dropout=invalid_input)

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

    def test_invalid_input_epoch(self):
        r"""Raise when input `epoch` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `epoch` is '
            'invalid.'
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
                BaseConfig(dataset='test', epoch=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`epoch` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`epoch` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_experiment(self):
        r"""Raise when input `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `experiment` '
            'is invalid.'
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
                BaseConfig(dataset='test', experiment=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must not be empty.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_is_uncased(self):
        r"""Raise when input `is_uncased` is invalid."""
        msg1 = 'Must raise `TypeError` when input `is_uncased` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j,
            1j, '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    is_uncased=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`is_uncased` must be an instance of `bool`.',
                msg=msg2
            )

    def test_invalid_input_learning_rate(self):
        r"""Raise when input `learning_rate` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`learning_rate` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, -1.0, math.nan, -math.nan, -math.inf, 0j, 1j, '', b'', [], (),
            {}, set(), object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    learning_rate=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`learning_rate` must be an instance of `float`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`learning_rate` must be bigger than `0.0`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_max_norm(self):
        r"""Raise when input `max_norm` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `max_norm` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, -1.0, math.nan, -math.nan, -math.inf, 0j, 1j, '', b'', [], (),
            {}, set(), object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    max_norm=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_norm` must be an instance of `float`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_norm` must be bigger than `0.0`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_max_seq_len(self):
        r"""Raise when input `max_seq_len` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `max_seq_len` '
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    max_seq_len=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_seq_len` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_seq_len` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_min_count(self):
        r"""Raise when input `min_count` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `min_count` is '
            'invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    min_count=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`min_count` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`min_count` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_model_class(self):
        r"""Raise when input `model_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `model_class` '
            'is invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    model_class=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model_class` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`model_class` must not be empty.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_num_linear_layers(self):
        r"""Raise when input `num_linear_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`num_linear_layers` is invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    num_linear_layers=invalid_input
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
        r"""Raise when input `num_rnn_layers` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`num_rnn_layers` is invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    num_rnn_layers=invalid_input
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

    def test_invalid_input_optimizer_class(self):
        r"""Raise when input `optimizer_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`optimizer_class` is invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    optimizer_class=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`optimizer_class` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`optimizer_class` must not be empty.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_seed(self):
        r"""Raise when input `seed` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `seed` is '
            'invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    seed=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`seed` must be an instance of `int`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`seed` must be bigger than or equal to `1`.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_invalid_input_tokenizer_class(self):
        r"""Raise when input `tokenizer_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`tokenizer_class` is invalid.'
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
                BaseConfig(
                    dataset='test',
                    experiment='test',
                    tokenizer_class=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokenizer_class` must be an instance of `str`.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, ValueError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokenizer_class` must not be empty.',
                    msg=msg2
                )
            else:
                self.fail(msg=msg1)

    def test_instance_attributes(self):
        r"""Declare required instance attributes."""
        msg1 = 'Missing instance attribute `{}`.'
        msg2 = 'Instance attribute `{}` must be an instance of `{}`.'
        msg3 = 'Instance attribute `{}` must be `{}`.'

        examples = (
            (
                ('batch_size', 111),
                ('checkpoint_step', 222),
                ('d_emb', 333),
                ('d_hid', 444),
                ('dataset', 'hello'),
                ('dropout', 0.42069),
                ('epoch', 555),
                ('experiment', 'world'),
                ('is_uncased', True),
                ('learning_rate', 0.69420),
                ('max_norm', 6.9),
                ('max_seq_len', 666),
                ('min_count', 777),
                ('model_class', 'HELLO'),
                ('num_linear_layers', 888),
                ('num_rnn_layers', 999),
                ('optimizer_class', 'WORLD'),
                ('seed', 101010),
                ('tokenizer_class', 'hello world'),
            ),
            (
                ('batch_size', 101010),
                ('checkpoint_step', 999),
                ('d_emb', 888),
                ('d_hid', 777),
                ('dataset', 'world'),
                ('dropout', 0.69420),
                ('epoch', 666),
                ('experiment', 'hello'),
                ('is_uncased', True),
                ('learning_rate', 0.42069),
                ('max_norm', 4.20),
                ('max_seq_len', 555),
                ('min_count', 444),
                ('model_class', 'hello world'),
                ('num_linear_layers', 333),
                ('num_rnn_layers', 222),
                ('optimizer_class', 'WORLD'),
                ('seed', 111),
                ('tokenizer_class', 'HELLO'),
            ),
        )

        for parameters in examples:
            pos = []
            kwargs = {}
            for attr, attr_val in parameters:
                pos.append(attr_val)
                kwargs[attr] = attr_val

            # Construct using positional and keyword arguments.
            configs = [
                BaseConfig(*pos),
                BaseConfig(**kwargs),
            ]

            for config in configs:
                for attr, attr_val in parameters:
                    self.assertTrue(
                        hasattr(config, attr),
                        msg=msg1.format(attr)
                    )
                    self.assertIsInstance(
                        getattr(config, attr),
                        type(attr_val),
                        msg=msg2.format(attr, type(attr_val).__name__)
                    )

                    self.assertEqual(
                        getattr(config, attr),
                        attr_val,
                        msg=msg3.format(attr, attr_val)
                    )


if __name__ == '__main__':
    unittest.main()
