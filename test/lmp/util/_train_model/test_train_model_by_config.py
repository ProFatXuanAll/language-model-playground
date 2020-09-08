r"""Test `lmp.util.train_model_by_config.`.

Usage:
    python -m unittest test.lmp.util._train_model.test_train_model_by_config
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
import lmp.tokenizer
import lmp.util


class TestTrainModelByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.train_model_by_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.dataset = 'I-AM-A-TEST-DATASET'
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.train_parameters = {
            'batch_size': [1, 2],
            'checkpoint_step': [1, 2],
            'epoch': [1, 2],
            'max_norm': [1.0, 2.0],
            'max_seq_len': [-1, 5],
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
        del cls.dataset
        del cls.experiment
        del cls.test_dir
        del cls.test_log_dir
        del cls.train_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""

        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            checkpoint_step=1,
            dataset=self.__class__.dataset,
            experiment=self.__class__.experiment,
            model_class='rnn',
            optimizer_class='sgd',
            tokenizer_class='char_dict'
        )
        self.dataset = lmp.dataset.LanguageModelDataset([''])
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()
        self.model = lmp.model.BaseRNNModel(
            d_emb=1,
            d_hid=1,
            dropout=0.1,
            num_rnn_layers=1,
            num_linear_layers=1,
            pad_token_id=0,
            vocab_size=self.tokenizer.vocab_size
        )
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=1e-4
        )

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.checkpoint
        del self.config
        del self.dataset
        del self.model
        del self.optimizer
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.train_model_by_config),
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
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.dataset.LanguageModelDataset,
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
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
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
                lmp.util.train_model_by_config(
                    checkpoint=invalid_input,
                    config=self.config,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
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
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`config` must be an instance of `lmp.config.BaseConfig`.',
                msg=msg2
            )

    def test_invalid_input_dataset(self):
        r"""Raise `TypeError` when input `dataset` is invalid."""
        msg1 = 'Must raise `TypeError` when input `dataset` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=invalid_input,
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`dataset` must be an instance of `lmp.dataset.LanguageModelDataset`.',
                msg=msg2)

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
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=self.dataset,
                    model=invalid_input,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer
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
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=invalid_input,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`optimizer` must be an instance of '
                '`Union[torch.optim.SGD, torch.optim.Adam]`.',
                msg=msg2
            )

    def test_invalid_input_tokenizer(self):
        r"""Raise `TypeError` when input `tokenizer` is invalid."""
        msg1 = 'Must raise `TypeError` when input `tokenizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.train_model_by_config(
                    checkpoint=self.checkpoint,
                    config=self.config,
                    dataset=self.dataset,
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
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
                max_seq_len,
                (model_cstr, optimizer_cstr, tokenizer),
        ) in product(*self.__class__.train_parameters.values()):
            config = lmp.config.BaseConfig(
                batch_size=batch_size,
                checkpoint_step=checkpoint_step,
                dataset=self.__class__.dataset,
                epoch=epoch,
                experiment=self.__class__.experiment,
                max_norm=max_norm,
                max_seq_len=max_seq_len
            )
            dataset = lmp.dataset.LanguageModelDataset([''] * batch_size)
            model = model_cstr(
                d_emb=1,
                d_hid=1,
                dropout=0.0,
                num_linear_layers=1,
                num_rnn_layers=1,
                pad_token_id=0,
                vocab_size=tokenizer.vocab_size
            ).to(config.device)
            optimizer = optimizer_cstr(
                params=model.parameters(),
                lr=1e-4
            )

            try:
                # Create test file.
                lmp.util.train_model_by_config(
                    checkpoint=-1,
                    config=config,
                    dataset=dataset,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer
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
                max_seq_len,
                (model_cstr, optimizer_cstr, tokenizer),
        ) in product(*self.__class__.train_parameters.values()):
            config = lmp.config.BaseConfig(
                batch_size=batch_size,
                checkpoint_step=checkpoint_step,
                dataset=self.__class__.dataset,
                epoch=epoch,
                experiment=self.__class__.experiment,
                max_norm=max_norm,
                max_seq_len=max_seq_len
            )
            dataset = lmp.dataset.LanguageModelDataset([''] * batch_size)
            model = model_cstr(
                d_emb=1,
                d_hid=1,
                dropout=0.0,
                num_linear_layers=1,
                num_rnn_layers=1,
                pad_token_id=0,
                vocab_size=tokenizer.vocab_size
            ).to(config.device)
            optimizer = optimizer_cstr(
                params=model.parameters(),
                lr=1e-4
            )

            try:
                # Create test file.
                lmp.util.train_model_by_config(
                    checkpoint=-1,
                    config=config,
                    dataset=dataset,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer
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
                max_seq_len,
                (model_cstr, optimizer_cstr, tokenizer),
        ) in product(*self.__class__.train_parameters.values()):
            config = lmp.config.BaseConfig(
                batch_size=batch_size,
                checkpoint_step=checkpoint_step,
                dataset=self.__class__.dataset,
                epoch=epoch,
                experiment=self.__class__.experiment,
                max_norm=max_norm,
                max_seq_len=max_seq_len
            )
            dataset = lmp.dataset.LanguageModelDataset([''] * batch_size)
            model = model_cstr(
                d_emb=1,
                d_hid=1,
                dropout=0.0,
                num_linear_layers=1,
                num_rnn_layers=1,
                pad_token_id=0,
                vocab_size=tokenizer.vocab_size
            ).to(config.device)
            optimizer = optimizer_cstr(
                params=model.parameters(),
                lr=1e-4
            )

            try:
                # Create test file.
                lmp.util.train_model_by_config(
                    checkpoint=-1,
                    config=config,
                    dataset=dataset,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer
                )
                config.epoch = 2 * epoch
                lmp.util.train_model_by_config(
                    checkpoint=epoch,
                    config=config,
                    dataset=dataset,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer
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
