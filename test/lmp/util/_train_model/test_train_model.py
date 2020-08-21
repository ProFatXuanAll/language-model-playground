r"""Test `lmp.util.train_model.`.

Usage:
    python -m unittest test.lmp.util._train_model.test_train_model
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

from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp


class TestTrainModel(unittest.TestCase):
    r"""Test Case for `lmp.util.train_model`."""

    def setUp(self):
        r"""Set up parameters for `train_model`."""
        dataset = ['apple', 'banana', 'papaya']
        tokenizer = lmp.tokenizer.CharDictTokenizer()
        collate_fn = lmp.dataset.BaseDataset.create_collate_fn(
            tokenizer=tokenizer,
            max_seq_len=20
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=collate_fn
        )

        self.checkpoint = -1
        self.checkpoint_step = 200
        self.data_loader = data_loader
        self.device = torch.tensor([10]).device
        self.epoch = 2
        self.experiment = 'test_util_train_model'
        self.max_norm = 2.5
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
        self.vocab_size = tokenizer.vocab_size

    def tearDown(self):
        r"""Delete parameters for `train_model`."""
        del self.checkpoint
        del self.checkpoint_step
        del self.data_loader
        del self.device
        del self.epoch
        del self.experiment
        del self.max_norm
        del self.model
        del self.optimizer
        del self.vocab_size
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.train_model),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='checkpoint_step',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='data_loader',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.utils.data.DataLoader,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='device',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.device,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='epoch',
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
                        name='max_norm',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=float,
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
                        name='vocab_size',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
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
                lmp.util.train_model(
                    checkpoint=invalid_input,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
                )

    def test_invalid_input_checkpoint_step(self):
        r"""Raise when `checkpoint_step` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `checkpoint_step` is '
            'invalid.'
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=invalid_input,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint_step` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint_step` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_data_loader(self):
        r"""Raise when `data_loader` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `data_loader` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=invalid_input,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`data_loader` must be an instance of '
                    '`torch.utils.data.DataLoader`.',
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=invalid_input,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`device` must be an instance of `torch.device`.',
                    msg=msg2
                )

    def test_invalid_input_epoch(self):
        r"""Raise when `epoch` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `epoch` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1.0, 1.1, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j, '',
            b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                (TypeError, ValueError),
                msg=msg1
            ) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=invalid_input,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`epoch` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`epoch` must be bigger than or equal to `1`.',
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=invalid_input,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
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

    def test_invalid_input_max_norm(self):
        r"""Raise when `max_norm` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `max_norm` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0j, 1j, '', b'', [], (), {}, set(),
            object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                (TypeError, ValueError),
                msg=msg1
            ) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=invalid_input,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_norm` must be an instance of `float`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_norm` must be bigger than `0.0`.',
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=invalid_input,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=invalid_input,
                    vocab_size=self.vocab_size
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

    def test_invalid_input_vocab_size(self):
        r"""Raise when `vocab_size` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when `vocab_size` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            '', b'', [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                (TypeError, ValueError),
                msg=msg1
            ) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
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


if __name__ == '__main__':
    unittest.main()
