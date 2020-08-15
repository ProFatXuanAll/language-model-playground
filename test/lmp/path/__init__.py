r"""Test `lmp.path`.

Usage:
    python -m unittest test.lmp.path.__init__
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestPath(unittest.TestCase):
    r"""Test case for `lmp.path`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.path
            # pylint: enable=C0415

            self.assertTrue(inspect.ismodule(lmp.path), msg=msg)
        except ImportError:
            self.fail(msg=msg)


if __name__ == '__main__':
    unittest.main()
