r"""Test `lmp.util.train_tokenizer_by_config.`.

Usage:
    python -m unittest test.lmp.util._train_tokenizer.test_train_tokenizer_by_config
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

from itertools import product

# self-made modules

import lmp.config
import lmp.dataset
import lmp.tokenizer
import lmp.util


class TestTrainTokenizerByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.train_tokenizer_by_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Setup dynamic parameters."""
        cls.dataset = 'I-AM-A-TEST-DATASET'
        cls.experiment = 'I-AM-A-TEST-EXPERIMENT'
        cls.tokenizer_parameters = {
            'is_uncased': [False, True],
            'batch_sequences': [
                ['hello', 'hello world'],
                ['world', 'hello world'],
            ],
            'min_count': [1, 2],
            'tokenizer_cstr': [
                lmp.tokenizer.CharDictTokenizer,
                lmp.tokenizer.CharListTokenizer,
                lmp.tokenizer.WhitespaceDictTokenizer,
                lmp.tokenizer.WhitespaceListTokenizer,
                lmp.tokenizer.CharDictTokenizer,
                lmp.tokenizer.CharListTokenizer,
            ],
        }

    @classmethod
    def tearDownClass(cls):
        r"""Delete dynamic parameters."""
        del cls.dataset
        del cls.experiment
        del cls.tokenizer_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.config = lmp.config.BaseConfig(
            dataset=self.__class__.dataset,
            experiment=self.__class__.experiment,
            tokenizer_class='char_dict'
        )
        self.dataset = lmp.dataset.LanguageModelDataset([''])
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.config
        del self.dataset
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.train_tokenizer_by_config),
            inspect.Signature(
                parameters=[
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
                lmp.util.train_tokenizer_by_config(
                    config=invalid_input,
                    dataset=self.dataset,
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
                lmp.util.train_tokenizer_by_config(
                    config=self.config,
                    dataset=invalid_input,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`dataset` must be an instance of `lmp.dataset.LanguageModelDataset`.',
                msg=msg2)

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
                lmp.util.train_tokenizer_by_config(
                    config=self.config,
                    dataset=self.dataset,
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
                msg=msg2
            )

    def test_increase_vocab(self):
        r"""Increase vocabulary."""
        msg = 'Must increase vocabulary.'

        for (
                is_uncased,
                batch_sequences,
                min_count,
                tokenizer_cstr
        ) in product(*self.__class__.tokenizer_parameters.values()):
            config = lmp.config.BaseConfig(
                dataset=self.__class__.dataset,
                experiment=self.__class__.experiment,
                min_count=min_count
            )
            dataset = lmp.dataset.LanguageModelDataset(
                batch_sequences=batch_sequences)
            tokenizer = tokenizer_cstr(is_uncased=is_uncased)
            v1 = tokenizer.vocab_size

            lmp.util.train_tokenizer_by_config(
                config=config,
                dataset=dataset,
                tokenizer=tokenizer
            )

            self.assertGreater(tokenizer.vocab_size, v1, msg=msg)


if __name__ == '__main__':
    unittest.main()
