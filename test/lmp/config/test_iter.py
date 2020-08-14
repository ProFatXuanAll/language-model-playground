r"""Test `lmp.config.BaseConfig.__iter__`.

Usage:
    python -m unittest test/lmp/config/test_iter.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

from typing import Iterable
from typing import Generator
from typing import Tuple
from typing import Union

# self-made modules

from lmp.config import BaseConfig


class TestIter(unittest.TestCase):
    r"""Test case for `lmp.config.BaseConfig.__iter__`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseConfig.__iter__),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=Generator[
                    Tuple[str, Union[bool, float, int, str]],
                    None,
                    None
                ]
            ),
            msg=msg
        )

    def test_yield_value(self):
        r"""Is an iterable which yield attributes in order."""
        msg = 'Must be an iterable which yield attributes in order.'
        examples = (
            {
                'batch_size': 111,
                'checkpoint_step': 222,
                'd_emb': 333,
                'd_hid': 444,
                'dataset': 'hello',
                'dropout': 0.42069,
                'epoch': 555,
                'experiment': 'world',
                'is_uncased': True,
                'learning_rate': 0.69420,
                'max_norm': 6.9,
                'max_seq_len': 666,
                'min_count': 777,
                'model_class': 'HELLO',
                'num_linear_layers': 888,
                'num_rnn_layers': 999,
                'optimizer_class': 'WORLD',
                'seed': 101010,
                'tokenizer_class': 'hello world',
            },
            {
                'batch_size': 101010,
                'checkpoint_step': 999,
                'd_emb': 888,
                'd_hid': 777,
                'dataset': 'world',
                'dropout': 0.69420,
                'epoch': 666,
                'experiment': 'hello',
                'is_uncased': True,
                'learning_rate': 0.42069,
                'max_norm': 4.20,
                'max_seq_len': 555,
                'min_count': 444,
                'model_class': 'hello world',
                'num_linear_layers': 333,
                'num_rnn_layers': 222,
                'optimizer_class': 'WORLD',
                'seed': 111,
                'tokenizer_class': 'HELLO',
            },
        )

        for ans_attributes in examples:
            config = BaseConfig(**ans_attributes)

            self.assertIsInstance(config, Iterable, msg=msg)

            for attr_key, attr_value in config:
                self.assertIn(attr_key, ans_attributes, msg=msg)
                self.assertTrue(hasattr(config, attr_key), msg=msg)
                self.assertIsInstance(
                    getattr(config, attr_key),
                    type(ans_attributes[attr_key]),
                    msg=msg
                )
                self.assertIsInstance(
                    getattr(config, attr_key),
                    type(attr_value),
                    msg=msg
                )
                self.assertEqual(
                    getattr(config, attr_key),
                    ans_attributes[attr_key],
                    msg=msg
                )
                self.assertEqual(
                    getattr(config, attr_key),
                    attr_value,
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
