r"""Test `lmp.util._config.py`.
Usage:
    python -m unittest test.lmp.util._config.__init__
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestUtilConfig(unittest.TestCase):
    r"""Test case for `lmp.util._config.py`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.util._config
            # pylint: enable=C0415

            self.assertTrue(inspect.ismodule(lmp.util._config), msg=msg)
        except ImportError:
            self.fail(msg=msg)

    def test_module_attributes(self):
        r"""Declare required module attributes."""
        msg1 = 'Missing module attribute `{}`.'
        msg2 = 'Module attribute `{}` must be a function.'
        msg3 = 'Inconsistent module signature.'
        examples = ('load_config',)

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.util._config
            # pylint: enable=C0415

            for attr in examples:
                self.assertTrue(
                    hasattr(lmp.util._config, attr),
                    msg=msg1.format(attr)
                )
                self.assertTrue(
                    inspect.isfunction(getattr(lmp.util._config, attr)),
                    msg=msg2.format(attr)
                )
        except ImportError:
            self.fail(msg=msg3)


if __name__ == '__main__':
    unittest.main()