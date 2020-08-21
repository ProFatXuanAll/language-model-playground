r"""Test `lmp.util.load_dataset.`.

Usage:
    python -m unittest test/lmp/util/_dataset/test_load_dataset.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import json
import math
import os
import unittest

# self-made modules

import lmp
import lmp.config
import lmp.dataset
import lmp.path


class TestLoadDatasetByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.load_dataset_by_config`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_dataset),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='config',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.config.BaseConfig,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.dataset.BaseDataset
            ),
            msg=msg
        )


    def test_invalid_input_dataset(self):
        r"""Raise when `dataset` is invalid."""
        msg1 = 'Must raise `TypeError`  when `dataset` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_dataset_by_config(invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`config` must be an instance of `lmp.config.BaseConfig`.',
                    msg=msg2
                )


    def test_return_type(self):
        r"""Return `lmp.config.BaseDataset`."""
        msg = 'Must return `lmp.config.BaseDataset`.'

        examples = (
            {
                'batch_size': 111,
                'checkpoint_step': 222,
                'd_emb': 333,
                'd_hid': 444,
                'dataset': 'news_collection_title',
                'dropout': 0.42069,
                'epoch': 555,
                'experiment': 'util_load_dataset_by_config_unittest',
                'is_uncased': True,
                'learning_rate': 0.69420,
                'max_norm': 6.9,
                'max_seq_len': 666,
                'min_count': 777,
                'model_class': 'gru',
                'num_linear_layers': 888,
                'num_rnn_layers': 999,
                'optimizer_class': 'sgd',
                'seed': 101010,
                'tokenizer_class': 'char_dict',
            },
            {
                'batch_size': 101010,
                'checkpoint_step': 999,
                'd_emb': 888,
                'd_hid': 777,
                'dataset': 'news_collection_desc',
                'dropout': 0.69420,
                'epoch': 666,
                'experiment': 'util_load_dataset_by_config_unittest',
                'is_uncased': True,
                'learning_rate': 0.42069,
                'max_norm': 4.20,
                'max_seq_len': 555,
                'min_count': 444,
                'model_class': 'rnn',
                'num_linear_layers': 333,
                'num_rnn_layers': 222,
                'optimizer_class': 'adam',
                'seed': 111,
                'tokenizer_class': 'char_list',
            },
        )

        for args in examples:
            config = lmp.config.BaseConfig(**args)
            dataset = lmp.util.load_dataset_by_config(config)
            self.assertIsInstance(dataset, lmp.dataset.BaseDataset, msg=msg)

if __name__ == '__main__':
    unittest.main()