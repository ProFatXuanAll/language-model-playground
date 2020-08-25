r"""Test `lmp.util.load_dataset.`.

Usage:
    python -m unittest test.lmp.util._dataset.test_load_dataset_by_config
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import math
import unittest

from typing import Union

# self-made modules

import lmp.config
import lmp.dataset
import lmp.util


class TestLoadDatasetByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.load_dataset_by_config`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_dataset_by_config),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='config',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.config.BaseConfig,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=Union[lmp.dataset.LanguageModelDataset, lmp.dataset.AnalogyDataset]
            ),
            msg=msg
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
                lmp.util.load_dataset_by_config(invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`config` must be an instance of `lmp.config.BaseConfig`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `lmp.config.LanguageModelDataset`."""
        msg = 'Must return `lmp.config.LanguageModelDataset`.'

        examples = (
            lmp.config.BaseConfig(
                dataset='news_collection_title',
                experiment='util_load_dataset_by_config_unittest',
            ),
            lmp.config.BaseConfig(
                dataset='news_collection_desc',
                experiment='util_load_dataset_by_config_unittest',
            ),
        )

        for config in examples:
            dataset = lmp.util.load_dataset_by_config(config)
            self.assertIsInstance(dataset, lmp.dataset.LanguageModelDataset, msg=msg)


if __name__ == '__main__':
    unittest.main()
