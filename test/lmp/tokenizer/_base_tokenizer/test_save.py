r"""Test `lmp.tokenizer.BaseTokenizer.save`.

Usage:
    python -m unittest test/lmp/tokenizer/_base_tokenizer/test_save.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

# self-made modules

from lmp.tokenizer import BaseTokenizer


class TestSave(unittest.TestCase):
    r"""Test case for `lmp.tokenizer.BaseTokenizer.save`."""

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
                'method `reset_vocab` not implemented yet.',
                msg=msg2
            )


if __name__ == '__main__':
    unittest.main()
