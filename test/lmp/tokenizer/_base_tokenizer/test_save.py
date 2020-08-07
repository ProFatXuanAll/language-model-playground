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

    def test_invalid_input_experiment(self):
        r"""Raise when input is invalid."""
        msg1 = 'Must raise `TypeError` or `ValueError` when input is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, '',
            b'', True, False, 0j, 1j, [], (), {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...,
        )

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseTokenizer):
            r"""Tricky skipping `reset_vocab`."""

            def reset_vocab(self):
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

    def test_abstract_method(self):
        r"""Raise `NotImplementedError` when subclass did not implement."""
        msg1 = (
            'Must raise `NotImplementedError` when subclass did not implement.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            (True, 'i-am-test'),
            (False, 'i-am-test'),
            (True, 'I-AM-TEST'),
            (False, 'I-AM-TEST'),
        )

        # pylint: disable=W0223
        # pylint: disable=W0231
        class SubClassTokenizer(BaseTokenizer):
            r"""Intented to not implement `reset_vocab`."""

            def __init__(self, is_uncased: bool = False):
                pass
        # pylint: enable=W0231
        # pylint: enable=W0223

        for is_uncased, experiment in examples:
            with self.assertRaises(NotImplementedError, msg=msg1) as ctx_man:
                SubClassTokenizer(is_uncased=is_uncased).save(
                    experiment=experiment
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                'In class `SubClassTokenizer`: '
                'function `reset_vocab` not implemented yet.',
                msg=msg2
            )


if __name__ == '__main__':
    unittest.main()
