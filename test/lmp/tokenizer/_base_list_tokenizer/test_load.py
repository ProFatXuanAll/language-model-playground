r"""Test `lmp.tokenizer.BaseListTokenizer.load`.

Usage:
    python -m unittest test.lmp.tokenizer._base_list_tokenizer.test_load
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import json
import math
import os
import unittest

# self-made modules

from lmp.path import DATA_PATH
from lmp.tokenizer import BaseListTokenizer


class TestLoad(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseListTokenizer.load`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory."""
        cls.experiment = 'I-AM-A-TEST'
        cls.test_dir = os.path.join(DATA_PATH, cls.experiment)
        if os.path.exists(cls.test_dir):
            for tokenizer_file in os.listdir(cls.test_dir):
                os.remove(os.path.join(cls.test_dir, tokenizer_file))
            os.removedirs(cls.test_dir)
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Clean up test directory."""
        os.removedirs(cls.test_dir)
        del cls.test_dir
        del cls.experiment
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseListTokenizer.load),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=inspect.Signature.empty
            ),
            msg=msg
        )

    def test_invalid_input_experiment(self):
        r"""Raise exception when input `experiment` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `experiment` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...,
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                BaseListTokenizer.load(experiment=invalid_input)

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

    def test_experiment_does_not_exist(self):
        r"""Raise `FileNotFoundError` when `experiment` does not exist."""
        msg1 = (
            'Must raise `FileNotFoundError` when `experiment` does not exist.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (self.__class__.experiment, 'I-AM-A-TEST-AND-I-DONT-EXIST')

        for experiment in examples:
            with self.assertRaises(FileNotFoundError, msg=msg1) as ctx_man:
                BaseListTokenizer.load(experiment=experiment)

            test_path = os.path.join(DATA_PATH, experiment, 'tokenizer.json')
            self.assertEqual(
                ctx_man.exception.args[0],
                f'File {test_path} does not exist.',
                msg=msg2
            )

    def test_load_result(self):
        r"""Load `tokenizer.json`."""
        msg = 'Inconsistent `tokenizer.json` format.'
        examples = (
            {
                'is_uncased': False,
                'token_to_id': {
                    'A': 0,
                    'B': 1,
                    'C': 2,
                },
            },
            {
                'is_uncased': True,
                'token_to_id': {
                    'a': 0,
                    'b': 1,
                    'c': 2,
                },
            },
        )

        test_path = os.path.join(self.__class__.test_dir, 'tokenizer.json')

        for obj in examples:
            try:
                # Create test file.
                with open(test_path, 'w', encoding='utf-8') as output_file:
                    json.dump(obj, output_file)

                tokenizer = BaseListTokenizer.load(
                    experiment=self.__class__.experiment
                )

                self.assertIsInstance(tokenizer, BaseListTokenizer, msg=msg)

                for attr_key, attr_value in obj.items():
                    self.assertTrue(hasattr(tokenizer, attr_key), msg=msg)
                    self.assertIsInstance(
                        getattr(tokenizer, attr_key),
                        type(attr_value),
                        msg=msg
                    )
                    self.assertEqual(
                        getattr(tokenizer, attr_key),
                        attr_value,
                        msg=msg
                    )
            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
