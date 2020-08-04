r"""Test `lmp.tokenizer.CharDictTokenizer.load`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_dict_tokenizer/test_load.py
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
from lmp.tokenizer import CharDictTokenizer


class TestLoad(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.CharDictTokenizer.load`."""

    def setUp(self):
        r"""Setup both cased and uncased tokenizer instances."""
        self.cased_tokenizer = CharDictTokenizer()
        self.uncased_tokenizer = CharDictTokenizer(is_uncased=True)
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
            inspect.signature(CharDictTokenizer.load),
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
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, '', b'', True, False,
            [], (), {}, set(), object(), lambda x: x, type, None,
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
                        '`experiment` must be instance of `str`.',
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
            'iamtestiamtestiamtestiamtestiamtest',
            'iamtestiamtestiamtestiamtestiamtest',
            'IAMTESTIAMTESTIAMTESTIAMTESTIAMTEST',
            'IAMTESTIAMTESTIAMTESTIAMTESTIAMTEST',
        )

        for experiment in examples:
            for tokenizer in self.tokenizers:
                tokenizer.save(experiment)

                file_dir = os.path.join(
                    DATA_PATH,
                    experiment
                )
                file_path = os.path.join(
                    file_dir,
                    'tokenizer.json'
                )
                with open(file_path, 'r', encoding='utf-8') as input_file:
                    obj = json.load(input_file)

                token_to_id = obj['token_to_id']
                id_to_token = {v: i for i, v in token_to_id.items()}
                self.assertEqual(
                    tokenizer.is_uncased,
                    obj['is_uncased'],
                    msg=msg
                )
                self.assertEqual(
                    tokenizer.token_to_id,
                    token_to_id,
                    msg=msg
                )
                self.assertEqual(
                    tokenizer.id_to_token,
                    id_to_token,
                    msg=msg
                )

                # Clean up test case.
                os.remove(file_path)
                os.removedirs(file_dir)


if __name__ == '__main__':
    unittest.main()
