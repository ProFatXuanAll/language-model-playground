r"""Test `lmp.path.DATA_PATH`.

Usage:
    python -m unittest test/lmp/path/test_data_path.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

# self-made modules

import lmp.path


class TestDataPath(unittest.TestCase):
    r"""Test case for `lmp.path.DATA_PATH`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent variable signature.'

        self.assertIsInstance(lmp.path.DATA_PATH, str, msg=msg)

    def test_is_absolute_path(self):
        r"""`DATA_PATH` is an absolute path."""
        msg = '`DATA_PATH` must be an absolute path.'

        self.assertTrue(os.path.isabs(lmp.path.DATA_PATH), msg=msg)

    def test_is_parent_path_project_root(self):
        r"""`DATA_PATH`'s parent path is `PROJECT_ROOT`."""
        msg = "`DATA_PATH`'s parent path must be `PROJECT_ROOT`."

        self.assertEqual(
            os.path.dirname(lmp.path.DATA_PATH),
            lmp.path.PROJECT_ROOT,
            msg=msg
        )

    def test_is_directory(self):
        r"""`DATA_PATH` is a directory if exists."""
        msg = '`DATA_PATH` must be a directory.'

        if os.path.exists(lmp.path.DATA_PATH):
            self.assertTrue(os.path.isdir(lmp.path.DATA_PATH), msg=msg)


if __name__ == '__main__':
    unittest.main()
