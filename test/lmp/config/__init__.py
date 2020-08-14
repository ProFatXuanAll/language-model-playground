r"""Test `lmp.config.py`.

Usage:
    python -m unittest test/lmp/config/__init__.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestConfig(unittest.TestCase):
    r"""Test case for `lmp.config.py`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.config
            # pylint: enable=C0415

            self.assertTrue(inspect.ismodule(lmp.config), msg=msg)
        except ImportError:
            self.fail(msg=msg)

    def test_module_attributes(self):
        r"""Declare required module attributes."""
        msg1 = 'Missing module attribute `{}`.'
        msg2 = 'Module attribute `{}` must be a class.'
        msg3 = 'Inconsistent module signature.'
        examples = ('BaseConfig',)

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.config
            # pylint: enable=C0415

            for attr in examples:
                self.assertTrue(
                    hasattr(lmp.config, attr),
                    msg=msg1.format(attr)
                )
                self.assertTrue(
                    inspect.isclass(getattr(lmp.config, attr)),
                    msg=msg2.format(attr)
                )
        except ImportError:
            self.fail(msg=msg3)


if __name__ == '__main__':
    unittest.main()
