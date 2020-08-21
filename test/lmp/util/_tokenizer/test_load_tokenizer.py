r"""Test `lmp.util.load_tokenizer.`.

Usage:
    python -m unittest test.lmp.util._tokenizer.test_load_tokenizer
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

from typing import Iterator
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp
import lmp.config
import lmp.model
import lmp.path


class TestLoadTokenizer(unittest.TestCase):
    r"""Test Case for `lmp.util.load_tokenizer`."""

    @classmethod
    def setUpClass(cls):
        cls.is_uncased_range = [True, False]
        cls.tokenizer_class_range = [
            'char_dict',
            'char_list',
            'whitespace_dict',
            'whitespace_list'
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.is_uncased_range
        del cls.tokenizer_class_range
        gc.collect()

    def setUp(self):
        r"""Set up parameters for `load_tokenizer`."""
        self.checkpoint = -1
        self.experiment= 'util_tokenizer_load_tokenizer_unittest'
        self.is_uncased = True
        self.tokenizer_class = 'char_dict'

    def tearDown(self):
        r"""Delete parameters for `load_tokenizer`."""
        del self.checkpoint
        del self.experiment
        del self.is_uncased
        del self.tokenizer_class
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_tokenizer),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='checkpoint',
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
                        name='is_uncased',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=False
                    ),
                    inspect.Parameter(
                        name='tokenizer_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default='char_list'
                    )
                ],
                return_annotation=lmp.tokenizer.BaseTokenizer
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
                lmp.util.load_tokenizer(
                    checkpoint=invalid_input,
                    experiment=self.experiment,
                    is_uncased=self.is_uncased,
                    tokenizer_class=self.tokenizer_class
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`checkpoint` must be an instance of `int`.',
                    msg=msg2
                )

    def test_invalid_input_experiment(self):
        r"""Raise when `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `experiment` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_tokenizer(
                    checkpoint=self.checkpoint,
                    experiment=invalid_input,
                    is_uncased=self.is_uncased,
                    tokenizer_class=self.tokenizer_class
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`experiment` must be an instance of `str`.',
                    msg=msg2
                )

    def test_invalid_input_is_uncased(self):
        r"""Raise when `is_uncased` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `is_uncased` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j, 1j,
            object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_tokenizer(
                    checkpoint=self.checkpoint,
                    experiment=self.experiment,
                    is_uncased=invalid_input,
                    tokenizer_class=self.tokenizer_class
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`is_uncased` must be an instance of `bool`.',
                    msg=msg2
                )
    
    def test_invalid_input_tokenizer_class(self):
        r"""Raise when `tokenizer_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` when `tokenizer_class` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, [], (), {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                (TypeError, ValueError),
                msg=msg1
            ) as ctx_man:
                lmp.util.load_tokenizer(
                    checkpoint=self.checkpoint,
                    experiment=self.experiment,
                    is_uncased=self.is_uncased,
                    tokenizer_class=invalid_input
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`tokenizer_class` must be an instance of `str`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'`{invalid_input}` does not support.\nSupported options:' +
                    ''.join(list(map(
                        lambda option: f'\n\t--tokenizer {option}',
                        [
                            'char_dict',
                            'char_list',
                            'whitespace_dict',
                            'whitespace_list',
                        ]
                    ))),
                    msg=msg2
                )

    def test_return_type(self):
        r"""Return `lmp.tokenizer.BaseTokenizer`."""
        msg = (
            'Must return `lmp.tokenizer.BaseTokenizer`.'
        )
        examples = (
            (
                is_uncased,
                tokenizer_class,
            )
            for is_uncased in  self.__class__.is_uncased_range
            for tokenizer_class in self.__class__.tokenizer_class_range
        )

        for is_uncased, tokenizer_class in examples:
            tokenizer = lmp.util.load_tokenizer(
                checkpoint=-1,
                experiment='util_load_tokenizer_unittest',
                is_uncased=is_uncased,
                tokenizer_class=tokenizer_class
            )

            self.assertIsInstance(
                tokenizer,
                lmp.tokenizer.BaseTokenizer,
                msg=msg
            )
            
if __name__ == '__main__':
    unittest.main()
