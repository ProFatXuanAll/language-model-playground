r"""Test `lmp.util._train_model.py`.
Usage:
    python -m unittest test.lmp.util._train_model.__init__
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestUtilTrainModel(unittest.TestCase):
    r"""Test case for `lmp.util._train_model.py`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.util._train_model
            # pylint: enable=C0415

            self.assertTrue(inspect.ismodule(lmp.util._train_model), msg=msg)
        except ImportError:
            self.fail(msg=msg)

    def test_module_attributes(self):
        r"""Declare required module attributes."""
        msg1 = 'Missing module attribute `{}`.'
        msg2 = 'Module attribute `{}` must be a function.'
        msg3 = 'Inconsistent module signature.'
        examples = (
            'train_model',
            'train_model_by_config',
        )

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.util._train_model
            # pylint: enable=C0415

            for attr in examples:
                self.assertTrue(
                    hasattr(lmp.util._train_model, attr),
                    msg=msg1.format(attr)
                )
                self.assertTrue(
                    inspect.isfunction(getattr(lmp.util._train_model, attr)),
                    msg=msg2.format(attr)
                )
        except ImportError:
            self.fail(msg=msg3)


if __name__ == '__main__':
    unittest.main()
