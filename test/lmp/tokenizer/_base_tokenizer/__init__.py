r"""Test `lmp.tokenizer._base_tokenizer.py`.

Usage:
    python -m unittest test/lmp/tokenizer/_base_tokenizer/__init__.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestBaseTokenizer(unittest.TestCase):
    r"""Test case for `lmp.tokenizer._base_tokenizer.py`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.tokenizer
            import lmp.tokenizer._base_tokenizer
            # pylint: enable=C0415

            # pylint: disable=W0212
            self.assertTrue(
                inspect.ismodule(lmp.tokenizer._base_tokenizer),
                msg=msg
            )
            # pylint: enable=W0212
        except ImportError:
            self.fail(msg=msg)


if __name__ == '__main__':
    unittest.main()
