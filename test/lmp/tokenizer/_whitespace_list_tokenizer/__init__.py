r"""Test `lmp.tokenizer._whitespace_list_tokenizer.py`.

Usage:
    python -m unittest \
        test/lmp/tokenizer/_whitespace_list_tokenizer/__init__.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestWhitespaceListTokenizer(unittest.TestCase):
    r"""Test case for `lmp.tokenizer._whitespace_list_tokenizer.py`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.tokenizer
            import lmp.tokenizer._whitespace_list_tokenizer
            # pylint: enable=C0415

            # pylint: disable=W0212
            self.assertTrue(
                inspect.ismodule(lmp.tokenizer._whitespace_list_tokenizer),
                msg=msg
            )
            # pylint: enable=W0212
        except ImportError:
            self.fail(msg=msg)

    def test_module_attributes(self):
        r"""Declare required module attributes."""
        msg1 = 'Missing module attribute `{}`.'
        msg2 = 'Module attribute `{}` must be a class.'
        msg3 = 'Inconsistent module signature.'
        examples = ('WhitespaceListTokenizer',)

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.tokenizer
            import lmp.tokenizer._whitespace_list_tokenizer
            # pylint: enable=C0415

            for attr in examples:
                self.assertTrue(
                    hasattr(lmp.tokenizer._whitespace_list_tokenizer, attr),
                    msg=msg1.format(attr)
                )
                self.assertTrue(
                    inspect.isclass(getattr(
                        lmp.tokenizer._whitespace_list_tokenizer,
                        attr
                    )),
                    msg=msg2.format(attr)
                )
        except ImportError:
            self.fail(msg=msg3)


if __name__ == '__main__':
    unittest.main()
