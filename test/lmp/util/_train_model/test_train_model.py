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

from itertools import product
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.model
import lmp.tokenizer
import lmp.util


class TestTrainModel(unittest.TestCase):
    r"""Test case for `lmp.util.train_model`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.train_parameters = {
            'batch_size': [1, 2],
            'checkpoint_step': [1, 2],
            'epoch': [1, 2],
            'max_norm': [1.0, 2.0],
            'train': [
                (
                    lmp.model.BaseRNNModel,
                    torch.optim.SGD,
                    lmp.tokenizer.CharDictTokenizer(is_uncased=True),
                ),
                (
                    lmp.model.GRUModel,
                    torch.optim.Adam,
                    lmp.tokenizer.CharListTokenizer(is_uncased=True),
                ),
                (
                    lmp.model.LSTMModel,
                    torch.optim.SGD,
                    lmp.tokenizer.WhitespaceDictTokenizer(is_uncased=True),
                ),
                (
                    lmp.model.BaseResRNNModel,
                    torch.optim.Adam,
                    lmp.tokenizer.WhitespaceListTokenizer(is_uncased=False),
                ),
                (
                    lmp.model.ResGRUModel,
                    torch.optim.SGD,
                    lmp.tokenizer.CharDictTokenizer(is_uncased=False),
                ),
                (
                    lmp.model.ResLSTMModel,
                    torch.optim.Adam,
                    lmp.tokenizer.CharListTokenizer(is_uncased=False),
                ),
            ],
            'vocab_size': [5, 10],
        }
        cls.test_dir = os.path.join(lmp.path.DATA_PATH, cls.experiment)
        cls.test_log_dir = os.path.join(
            lmp.path.DATA_PATH,
            'log',
            cls.experiment
        )
        os.makedirs(cls.test_dir)
        os.makedirs(cls.test_log_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory and delete dynamic parameters."""
        os.removedirs(cls.test_dir)
        os.removedirs(cls.test_log_dir)
        del cls.experiment
        del cls.test_dir
        del cls.test_log_dir
        del cls.train_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.checkpoint = -1
        self.checkpoint_step = 1
        self.data_loader = torch.utils.data.DataLoader(
            [''],
            batch_size=1,
            shuffle=True
        )
        self.device = torch.device('cpu')
        self.epoch = 1
        self.max_norm = 1.0
        self.model = lmp.model.BaseRNNModel(
            d_emb=1,
            d_hid=1,
            dropout=0.0,
            num_rnn_layers=1,
            num_linear_layers=1,
            pad_token_id=0,
            vocab_size=5
        )
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=1e-4
        )
        self.vocab_size = 5

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.checkpoint
        del self.checkpoint_step
        del self.data_loader
        del self.device
        del self.epoch
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
                return_annotation=None
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
                lmp.util.train_model(
                    checkpoint=invalid_input,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.__class__.experiment,
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
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be bigger than or equal to `-1`.',
                    msg=msg2
                )

    def test_invalid_input_checkpoint_step(self):
        r"""Raise exception when input `checkpoint_step` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`checkpoint_step` is invalid.'
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=invalid_input,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.__class__.experiment,
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
        r"""Raise `TypeError` when input `data_loader` is invalid."""
        msg1 = 'Must raise `TypeError` when input `data_loader` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 1, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=invalid_input,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.__class__.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`data_loader` must be an instance of '
                '`torch.utils.data.DataLoader`.',
                msg=msg2
            )

    def test_invalid_input_device(self):
        r"""Raise `TypeError` when input `device` is invalid."""
        msg1 = 'Must raise `TypeError` when input `device` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 1, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=invalid_input,
                    epoch=self.epoch,
                    experiment=self.__class__.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`device` must be an instance of `torch.device`.',
                msg=msg2
            )

    def test_invalid_input_epoch(self):
        r"""Raise exception when input `epoch` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `epoch` is '
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
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=invalid_input,
                    experiment=self.__class__.experiment,
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
        r"""Raise excpetion when input `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `experiment` '
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
        r"""Raise exception when input `max_norm` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `max_norm` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, -1.0, math.nan, -math.nan, -math.inf, 0j,
            1j, '', b'', (), [], {}, set(), object(), lambda x: x, type, None,
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
                    experiment=self.__class__.experiment,
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
        r"""Raise `TypeError` when input `model` is invalid."""
        msg1 = 'Must raise `TypeError` when input `model` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 1, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.__class__.experiment,
                    max_norm=self.max_norm,
                    model=invalid_input,
                    optimizer=self.optimizer,
                    vocab_size=self.vocab_size
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`model` must be an instance of '
                '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.',
                msg=msg2
            )

    def test_invalid_input_optimizer(self):
        r"""Raise `TypeError` when input `optimizer` is invalid."""
        msg1 = 'Must raise `TypeError` when input `optimizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 1, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model(
                    checkpoint=self.checkpoint,
                    checkpoint_step=self.checkpoint_step,
                    data_loader=self.data_loader,
                    device=self.device,
                    epoch=self.epoch,
                    experiment=self.__class__.experiment,
                    max_norm=self.max_norm,
                    model=self.model,
                    optimizer=invalid_input,
                    vocab_size=self.vocab_size
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`optimizer` must be an instance of '
                '`Union[torch.optim.SGD, torch.optim.Adam]`.',
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
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', (), [], {}, set(), object(), lambda x: x, type,
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
                    experiment=self.__class__.experiment,
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

    def test_save_checkpoint(self):
        r"""Save checkpoint at each `checkpoint_step`."""
        msg = 'Must save checkpoint at each `checkpoint_step`.'

        for (
                batch_size,
                checkpoint_step,
                epoch,
                max_norm,
                (model_cstr, optimizer_cstr, tokenizer),
                vocab_size
        ) in product(*self.__class__.train_parameters.values()):
            data_loader = torch.utils.data.DataLoader(
                batch_size=batch_size,
                dataset=lmp.dataset.BaseDataset([''] * batch_size),
                collate_fn=lmp.dataset.BaseDataset.create_collate_fn(
                    tokenizer=tokenizer,
                    max_seq_len=-1
                )
            )
            model = model_cstr(
                d_emb=1,
                d_hid=1,
                dropout=0.0,
                num_linear_layers=1,
                num_rnn_layers=1,
                pad_token_id=0,
                vocab_size=vocab_size
            )
            optimizer = optimizer_cstr(
                params=model.parameters(),
                lr=1e-4
            )

            try:
                # Create test file.
                lmp.util.train_model(
                    checkpoint=-1,
                    checkpoint_step=checkpoint_step,
                    data_loader=data_loader,
                    device=torch.device('cpu'),
                    epoch=epoch,
                    experiment=self.__class__.experiment,
                    max_norm=max_norm,
                    model=model,
                    optimizer=optimizer,
                    vocab_size=vocab_size
                )

                for ckpt in range(0, epoch, checkpoint_step):
                    if ckpt == 0:
                        continue
                    self.assertTrue(
                        os.path.exists(os.path.join(
                            self.__class__.test_dir,
                            f'model-{ckpt}.pt'
                        )),
                        msg=msg
                    )
                    self.assertTrue(
                        os.path.exists(os.path.join(
                            self.__class__.test_dir,
                            f'optimizer-{ckpt}.pt'
                        )),
                        msg=msg
                    )
            finally:
                # Clean up test file.
                for ckpt in os.listdir(self.__class__.test_dir):
                    os.remove(os.path.join(self.__class__.test_dir, ckpt))
                for log in os.listdir(self.__class__.test_log_dir):
                    os.remove(os.path.join(self.__class__.test_log_dir, log))

    def test_log_loss(self):
        r"""Log loss."""
        msg = 'Must log loss.'

        for (
                batch_size,
                checkpoint_step,
                epoch,
                max_norm,
                (model_cstr, optimizer_cstr, tokenizer),
                vocab_size
        ) in product(*self.__class__.train_parameters.values()):
            data_loader = torch.utils.data.DataLoader(
                batch_size=batch_size,
                dataset=lmp.dataset.BaseDataset([''] * batch_size),
                collate_fn=lmp.dataset.BaseDataset.create_collate_fn(
                    tokenizer=tokenizer,
                    max_seq_len=-1
                )
            )
            model = model_cstr(
                d_emb=1,
                d_hid=1,
                dropout=0.0,
                num_linear_layers=1,
                num_rnn_layers=1,
                pad_token_id=0,
                vocab_size=vocab_size
            )
            optimizer = optimizer_cstr(
                params=model.parameters(),
                lr=1e-4
            )

            try:
                # Create test file.
                lmp.util.train_model(
                    checkpoint=-1,
                    checkpoint_step=checkpoint_step,
                    data_loader=data_loader,
                    device=torch.device('cpu'),
                    epoch=epoch,
                    experiment=self.__class__.experiment,
                    max_norm=max_norm,
                    model=model,
                    optimizer=optimizer,
                    vocab_size=vocab_size
                )

                self.assertGreater(
                    len(os.listdir(self.__class__.test_log_dir)),
                    0,
                    msg=msg
                )
            finally:
                # Clean up test file.
                for ckpt in os.listdir(self.__class__.test_dir):
                    os.remove(os.path.join(self.__class__.test_dir, ckpt))
                for log in os.listdir(self.__class__.test_log_dir):
                    os.remove(os.path.join(self.__class__.test_log_dir, log))

    def test_keep_training(self):
        r"""Keep training from `checkpoint`."""
        msg = 'Must keep training from `checkpoint`.'

        for (
                batch_size,
                checkpoint_step,
                epoch,
                max_norm,
                (model_cstr, optimizer_cstr, tokenizer),
                vocab_size
        ) in product(*self.__class__.train_parameters.values()):
            data_loader = torch.utils.data.DataLoader(
                batch_size=batch_size,
                dataset=lmp.dataset.BaseDataset([''] * batch_size),
                collate_fn=lmp.dataset.BaseDataset.create_collate_fn(
                    tokenizer=tokenizer,
                    max_seq_len=-1
                )
            )
            model = model_cstr(
                d_emb=1,
                d_hid=1,
                dropout=0.0,
                num_linear_layers=1,
                num_rnn_layers=1,
                pad_token_id=0,
                vocab_size=vocab_size
            )
            optimizer = optimizer_cstr(
                params=model.parameters(),
                lr=1e-4
            )

            try:
                # Create test file.
                lmp.util.train_model(
                    checkpoint=-1,
                    checkpoint_step=checkpoint_step,
                    data_loader=data_loader,
                    device=torch.device('cpu'),
                    epoch=epoch,
                    experiment=self.__class__.experiment,
                    max_norm=max_norm,
                    model=model,
                    optimizer=optimizer,
                    vocab_size=vocab_size
                )
                lmp.util.train_model(
                    checkpoint=epoch,
                    checkpoint_step=checkpoint_step,
                    data_loader=data_loader,
                    device=torch.device('cpu'),
                    epoch=2 * epoch,
                    experiment=self.__class__.experiment,
                    max_norm=max_norm,
                    model=model,
                    optimizer=optimizer,
                    vocab_size=vocab_size
                )

                for ckpt in range(0, 2 * epoch, checkpoint_step):
                    if ckpt == 0:
                        continue
                    self.assertTrue(
                        os.path.exists(os.path.join(
                            self.__class__.test_dir,
                            f'model-{ckpt}.pt'
                        )),
                        msg=msg
                    )
                    self.assertTrue(
                        os.path.exists(os.path.join(
                            self.__class__.test_dir,
                            f'optimizer-{ckpt}.pt'
                        )),
                        msg=msg
                    )
            finally:
                # Clean up test file.
                for ckpt in os.listdir(self.__class__.test_dir):
                    os.remove(os.path.join(self.__class__.test_dir, ckpt))
                for log in os.listdir(self.__class__.test_log_dir):
                    os.remove(os.path.join(self.__class__.test_log_dir, log))


if __name__ == '__main__':
    unittest.main()
