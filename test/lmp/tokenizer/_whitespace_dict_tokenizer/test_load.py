r"""Test `lmp.tokenizer.WhitespaceDictTokenizer.load`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_dict_tokenizer/test_load.py
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
from lmp.tokenizer import WhitespaceDictTokenizer


class TestLoad(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.WhitespaceDictTokenizer.load`."""

    @classmethod
    def setUpClass(cls):
        r"""Create test directory."""
        cls.experiment = 'I-AM-A-TEST'
        cls.test_dir = os.path.join(DATA_PATH, cls.experiment)
        os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        r"""Clean up test directory."""
        os.removedirs(cls.test_dir)

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = WhitespaceDictTokenizer()
        self.uncased_tokenizer = WhitespaceDictTokenizer(is_uncased=True)
        self.tokenizers = [self.cased_tokenizer, self.uncased_tokenizer]

    def tearDown(self):
        r"""Delete both cased and uncased tokenizer instances."""
        del self.tokenizers
        del self.cased_tokenizer
        del self.uncased_tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(WhitespaceDictTokenizer.load),
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

    def test_invalid_input(self):
        r"""Raise `TypeError` or `ValueError` when input is invalid."""
        msg1 = 'Must raise `TypeError` or `ValueError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, '', b'', True, False, 0j, 1j,
            [], (), {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(
                        (TypeError, ValueError),
                        msg=msg1
                ) as ctx_man:
                    tokenizer.load(invalid_input)

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

    def test_load_file(self):
        r"""Load `tokenizer.json`."""
        msg = 'Inconsistent `tokenizer.json` format.'
        examples = (
            ('is_uncased', bool),
            ('token_to_id', dict),
        )

        for tokenizer in self.tokenizers:
            test_path = os.path.join(self.__class__.test_dir, 'tokenizer.json')

            try:
                # Create test file.
                tokenizer.save(self.__class__.experiment)
                with open(test_path, 'r', encoding='utf-8') as input_file:
                    obj = json.load(input_file)

                for attr_key, attr_type in examples:
                    self.assertIn(attr_key, obj, msg=msg)
                    self.assertIsInstance(obj[attr_key], attr_type, msg=msg)
                    self.assertEqual(
                        obj[attr_key],
                        getattr(tokenizer, attr_key),
                        msg=msg
                    )
                id_to_token = {v: i for i, v in obj['token_to_id'].items()}
                self.assertEqual(
                    id_to_token,
                    tokenizer.id_to_token,
                    msg=msg
                )
            finally:
                # Clean up test case.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
