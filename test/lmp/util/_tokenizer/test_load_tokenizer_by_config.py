r"""Test `lmp.util.load_tokenizer_by_config.`.

Usage:
    python -m unittest test.lmp.util._tokenizer.test_load_tokenizer_by_config
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

import lmp.config
import lmp.util


class TestLoadTokenizer(unittest.TestCase):
    r"""Test case for `lmp.util.load_tokenizer_by_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory and setup dynamic parameters."""
        cls.checkpoint = 10
        cls.dataset = 'I-AM-A-TEST-DATASET'
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
        del cls.dataset
        del cls.experiment
        del cls.test_dir
        del cls.tokenizer_parameters
        gc.collect()

    def setUp(self):
        r"""Setup fixed parameters."""
        self.checkpoint = -1
        self.config = lmp.config.BaseConfig(
            dataset=self.__class__.dataset,
            experiment=self.__class__.experiment
        )

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.checkpoint
        del self.config
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.load_tokenizer_by_config),
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
                lmp.util.load_tokenizer_by_config(
                    checkpoint=invalid_input,
                    config=self.config
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
                lmp.util.load_tokenizer_by_config(
                    checkpoint=self.checkpoint,
                    config=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`config` must be an instance of `lmp.config.BaseConfig`.',
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
            config = lmp.config.BaseConfig(
                dataset=self.__class__.dataset,
                experiment=self.__class__.experiment,
                is_uncased=is_uncased,
                tokenizer_class=tokenizer_class
            )

            tokenizer_1 = lmp.util.load_tokenizer_by_config(
                checkpoint=-1,
                config=config
            )

            self.assertIsInstance(tokenizer_1, tokenizer_cstr, msg=msg)

            try:
                # Create test file.
                tokenizer_1.save(experiment=self.__class__.experiment)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                tokenizer_2 = lmp.util.load_tokenizer_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config
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
            config = lmp.config.BaseConfig(
                dataset=self.__class__.dataset,
                experiment=self.__class__.experiment,
                is_uncased=is_uncased,
                tokenizer_class=tokenizer_class
            )

            try:
                # Create test file.
                ans_tokenizer = tokenizer_cstr(is_uncased=is_uncased)
                ans_tokenizer.save(self.__class__.experiment)
                self.assertTrue(os.path.exists(test_path), msg=msg)

                tokenizer_1 = lmp.util.load_tokenizer_by_config(
                    checkpoint=-1,
                    config=config
                )
                tokenizer_2 = lmp.util.load_tokenizer_by_config(
                    checkpoint=self.__class__.checkpoint,
                    config=config
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
