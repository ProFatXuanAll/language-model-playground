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
import os
import unittest

from itertools import product

# self-made modules

import lmp.path
import lmp.util


class TestLoadTokenizer(unittest.TestCase):
    r"""Test case for `lmp.util.load_tokenizer`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.checkpoint = 10
        cls.experiment = 'I-AM-A-TEST-FOLDER'
        cls.tokenizer_parameters = {
            'is_uncased': [False, True],
            'tokenizer': [
                ('char_dict', lmp.tokenizer.CharDictTokenizer),
                ('char_list', lmp.tokenizer.CharListTokenizer),
                ('whitespace_dict', lmp.tokenizer.WhitespaceDictTokenizer),
                ('whitespace_list', lmp.tokenizer.WhitespaceListTokenizer),
            ],
        }
        cls.test_dir = os.path.join(lmp.path.DATA_PATH, cls.experiment)
        if os.path.exists(cls.test_dir):
            for tokenizer_file in os.listdir(cls.test_dir):
                os.remove(os.path.join(cls.test_dir, tokenizer_file))
            os.removedirs(cls.test_dir)
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Remove test directory and delete dynamic parameters."""
        os.removedirs(cls.test_dir)
        del cls.checkpoint
        del cls.experiment
        del cls.test_dir
        del cls.tokenizer_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters"""
        self.checkpoint = -1
        self.is_uncased = True
        self.tokenizer_class = 'char_dict'

    def tearDown(self):
        r"""Delete fixed parameters"""
        del self.checkpoint
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
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer_class',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=lmp.tokenizer.BaseTokenizer
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
                lmp.util.load_tokenizer(
                    checkpoint=invalid_input,
                    experiment=self.__class__.experiment,
                    is_uncased=self.is_uncased,
                    tokenizer_class=self.tokenizer_class
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

    def test_invalid_input_experiment(self):
        r"""Raise exception when input `experiment` is invalid."""
        msg1 = (
            'Must raise `FileNotFoundError`, `TypeError` or `ValueError` when '
            'input `experiment` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', 'I-DO-NOT-EXIST', b'', (), [], {}, set(),
            object(), lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (FileNotFoundError, TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.load_tokenizer(
                    checkpoint=0,
                    experiment=invalid_input,
                    is_uncased=self.is_uncased,
                    tokenizer_class=self.tokenizer_class
                )

            if isinstance(ctx_man.exception, FileNotFoundError):
                test_path = os.path.join(
                    lmp.path.DATA_PATH,
                    invalid_input,
                    'tokenizer.json'
                )
                self.assertEqual(
                    ctx_man.exception.args[0],
                    f'File {test_path} does not exist.',
                    msg=msg2
                )
            elif isinstance(ctx_man.exception, TypeError):
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

    def test_invalid_input_is_uncased(self):
        r"""Raise `TypeError` when input `is_uncased` is invalid."""
        msg1 = 'Must raise `TypeError` when input `is_uncased` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j,
            1j, '', b'', (), [], {}, object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.load_tokenizer(
                    checkpoint=self.checkpoint,
                    experiment=self.__class__.experiment,
                    is_uncased=invalid_input,
                    tokenizer_class=self.tokenizer_class
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`is_uncased` must be an instance of `bool`.',
                msg=msg2
            )

    def test_invalid_input_tokenizer_class(self):
        r"""Raise exception when input `tokenizer_class` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input '
            '`tokenizer_class` is invalid.'
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
                lmp.util.load_tokenizer(
                    checkpoint=self.checkpoint,
                    experiment=self.__class__.experiment,
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
                    f'tokenizer `{invalid_input}` does not support.\n' +
                    'Supported options:' +
                    ''.join(list(map(
                        lambda option: f'\n\t--tokenizer_class {option}',
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
        msg = 'Must return `lmp.tokenizer.BaseTokenizer`.'

        test_path = os.path.join(
            self.__class__.test_dir,
            'tokenizer.json'
        )

        for (
                is_uncased,
                (tokenizer_class, tokenizer_cstr)
        ) in product(*self.__class__.tokenizer_parameters.values()):
            tokenizer_1 = lmp.util.load_tokenizer(
                checkpoint=-1,
                experiment=self.__class__.experiment,
                is_uncased=is_uncased,
                tokenizer_class=tokenizer_class
            )

            self.assertIsInstance(tokenizer_1, tokenizer_cstr, msg=msg)

            try:
                # Create test file.
                tokenizer_1.save(experiment=self.__class__.experiment)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                tokenizer_2 = lmp.util.load_tokenizer(
                    checkpoint=self.__class__.checkpoint,
                    experiment=self.__class__.experiment,
                    is_uncased=is_uncased,
                    tokenizer_class=tokenizer_class
                )

                self.assertIsInstance(tokenizer_2, tokenizer_cstr, msg=msg)
            finally:
                # Clean up test file.
                os.remove(test_path)

    def test_load_result(self):
        r"""Load result must be consistent."""
        msg = 'Inconsistent load result.'

        test_path = os.path.join(
            self.__class__.test_dir,
            'tokenizer.json'
        )

        for (
                is_uncased,
                (tokenizer_class, tokenizer_cstr)
        ) in product(*self.__class__.tokenizer_parameters.values()):
            try:
                # Create test file.
                ans_tokenizer = tokenizer_cstr(is_uncased=is_uncased)
                ans_tokenizer.save(self.__class__.experiment)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                tokenizer_1 = lmp.util.load_tokenizer(
                    checkpoint=-1,
                    experiment=self.__class__.experiment,
                    is_uncased=is_uncased,
                    tokenizer_class=tokenizer_class
                )
                tokenizer_2 = lmp.util.load_tokenizer(
                    checkpoint=self.__class__.checkpoint,
                    experiment=self.__class__.experiment,
                    is_uncased=is_uncased,
                    tokenizer_class=tokenizer_class
                )

                self.assertEqual(
                    len(ans_tokenizer.token_to_id),
                    len(tokenizer_1.token_to_id),
                    msg=msg
                )
                self.assertEqual(
                    len(ans_tokenizer.token_to_id),
                    len(tokenizer_2.token_to_id),
                    msg=msg
                )

                self.assertEqual(
                    ans_tokenizer.token_to_id,
                    tokenizer_2.token_to_id,
                    msg=msg
                )

            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
