r"""Test `lmp.tokenizer.BaseTokenizer.save`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_base_tokenizer/test_save.py
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

from lmp.path import DATA_PATH
from lmp.tokenizer import BaseTokenizer


class TestSave(unittest.TestCase):
    r"""Test Case for `lmp.tokenizer.BaseTokenizer.save`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(BaseTokenizer.save),
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

    def test_invalid_input(self):
        r"""Raise `TypeError` or `ValueError` when input is invalid."""
        msg1 = 'Must raise `TypeError` or `ValueError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, math.inf, '', b'', True, False,
            [], (), {}, set(), object(), lambda x: x, type, None,
        )

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseTokenizer):
            r"""Intented to not implement `save`."""

            def __init__(self):
                pass
        # pylint: enable=W0231
        # pylint: enable=W0223

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                SubClassTokenizer().save(invalid_input)

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

    def test_output_file(self):
        r"""Create `tokenizer.json`."""
        msg1 = 'Must create `tokenizer.json`.'
        msg2 = 'Inconsistent `tokenizer.json` format.'
        examples = (
            (True, 'iamtestiamtestiamtestiamtestiamtest'),
            (False, 'iamtestiamtestiamtestiamtestiamtest'),
            (True, 'IAMTESTIAMTESTIAMTESTIAMTESTIAMTEST'),
            (False, 'IAMTESTIAMTESTIAMTESTIAMTESTIAMTEST'),
        )

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseTokenizer):
            r"""Intented to not implement `save`."""

            def reset_vocab(self):
                self.token_to_id = [1, 2, 3]
        # pylint: enable=W0231
        # pylint: enable=W0223

        for is_uncased, experiment in examples:
            SubClassTokenizer(is_uncased=is_uncased).save(experiment)

            file_dir = os.path.join(
                DATA_PATH,
                experiment
            )
            file_path = os.path.join(
                file_dir,
                'tokenizer.json'
            )

            self.assertTrue(os.path.exists(file_path), msg=msg1)
            with open(file_path, 'r') as input_file:
                obj = json.load(input_file)
            self.assertIn('is_uncased', obj, msg=msg2)
            self.assertEqual(obj['is_uncased'], is_uncased, msg=msg2)
            self.assertIn('token_to_id', obj, msg=msg2)
            self.assertEqual(obj['token_to_id'], [1, 2, 3], msg=msg2)

            # Clean up test case.
            os.remove(file_path)
            os.removedirs(file_dir)


if __name__ == '__main__':
    unittest.main()
