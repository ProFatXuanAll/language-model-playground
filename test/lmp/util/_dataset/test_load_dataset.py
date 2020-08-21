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


class TestLoadDataset(unittest.TestCase):
    r"""Test Case for `lmp.util.load_dataset`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_dataset),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='dataset',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.dataset.BaseDataset
            ),
            msg=msg
        )


    def test_invalid_input_dataset(self):
        r"""Raise when `dataset` is invalid."""
        msg1 = 'Must raise `TypeError` or `ValueError` when `dataset` ' 
        'is invalid.'
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
                lmp.util.load_dataset(invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`dataset` must be instance of `str`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'dataset `{invalid_input}` does not support.\nSupported options:' +
                    ''.join(list(map(
                        lambda option: f'\n\t--dataset {option}',
                        [
                            'news_collection_desc',
                            'news_collection_title',
                        ]
                    ))),
                    msg=msg2
                )

            

    def test_return_type(self):
        r"""Return `lmp.config.BaseDataset`."""
        msg = 'Must return `lmp.config.BaseDataset`.'

        examples = (
            'news_collection_desc',
            'news_collection_title',
        )

        for dataset_name in examples:
            dataset = lmp.util.load_dataset(dataset=dataset_name)
            self.assertIsInstance(dataset, lmp.dataset.BaseDataset, msg=msg)


if __name__ == '__main__':
    unittest.main()
