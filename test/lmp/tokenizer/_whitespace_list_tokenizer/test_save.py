r"""Test `lmp.tokenizer.WhitespaceListTokenizer.save`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_list_tokenizer/test_save.py
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
from lmp.tokenizer import WhitespaceListTokenizer


class TestSave(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.save`."""

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
        del cls.test_dir
        del cls.experiment
        gc.collect()

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = WhitespaceListTokenizer()
        self.uncased_tokenizer = WhitespaceListTokenizer(is_uncased=True)
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
            inspect.signature(WhitespaceListTokenizer.save),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='experiment',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                ],
                return_annotation=None
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
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...,
        )

        for invalid_input in examples:
            for tokenizer in self.tokenizers:
                with self.assertRaises(
                        (TypeError, ValueError),
                        msg=msg1
                ) as ctx_man:
                    tokenizer.save(experiment=invalid_input)

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

    def test_save_result(self):
        r"""Create `tokenizer.json`."""
        msg1 = 'Must create `tokenizer.json`.'
        msg2 = 'Inconsistent `tokenizer.json` format.'
        examples = ('is_uncased', 'token_to_id')

        test_path = os.path.join(self.__class__.test_dir, 'tokenizer.json')
        for tokenizer in self.tokenizers:
            try:
                # Create test file.
                tokenizer.save(experiment=self.__class__.experiment)
                self.assertTrue(os.path.exists(test_path), msg=msg1)

                with open(test_path, 'r') as input_file:
                    obj = json.load(input_file)

                for attr_key in examples:
                    self.assertIn(attr_key, obj, msg=msg2)
                    self.assertIsInstance(
                        obj[attr_key],
                        type(getattr(tokenizer, attr_key)),
                        msg=msg2
                    )
                    self.assertEqual(
                        obj[attr_key],
                        getattr(tokenizer, attr_key),
                        msg=msg2
                    )
            finally:
                # Clean up test file.
                os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
