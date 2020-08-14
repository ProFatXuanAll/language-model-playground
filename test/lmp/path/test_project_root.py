r"""Test `lmp.path.PROJECT_ROOT`.

Usage:
    python -m unittest test/lmp/path/test_project_root.py
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


class TestProjectRoot(unittest.TestCase):
    r"""Test case for `lmp.path.PROJECT_ROOT`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent variable signature.'

        self.assertIsInstance(lmp.path.PROJECT_ROOT, str, msg=msg)

    def test_is_absolute_path(self):
        r"""`PROJECT_ROOT` is an absolute path."""
        msg = '`PROJECT_ROOT` must be an absolute path.'

        self.assertTrue(os.path.isabs(lmp.path.PROJECT_ROOT), msg=msg)

    def test_is_directory(self):
        r"""`PROJECT_ROOT` is a directory."""
        msg = '`PROJECT_ROOT` must be a directory.'

        self.assertTrue(os.path.isdir(lmp.path.PROJECT_ROOT), msg=msg)

    def test_project_root_structure(self):
        r"""`PROJECT_ROOT` is the root of project."""
        msg = 'missing {} `{}` in project root.'
        examples = (
            ('file', '.gitignore', True),
            ('file', 'README.md', True),
            ('file', 'requirements.txt', True),
            ('file', 'run_generate.py', True),
            ('file', 'run_perplexity_evaluation.py', True),
            ('file', 'run_train.py', True),
            ('directory', 'data', False),
            ('directory', 'lmp', True),
            ('directory', 'test', True),
            ('directory', 'venv', False),
        )

        for file_type, file_name, is_required in examples:
            file_path = os.path.join(lmp.path.PROJECT_ROOT, file_name)
            if is_required or os.path.exists(file_path):
                self.assertTrue(
                    os.path.exists(file_path),
                    msg=msg.format(file_type, file_name)
                )
                if file_type == 'file':
                    self.assertTrue(
                        os.path.isfile(file_path),
                        msg=msg.format(file_type, file_name)
                    )
                elif file_type == 'directory':
                    self.assertTrue(
                        os.path.isdir(file_path),
                        msg=msg.format(file_type, file_name)
                    )
                else:
                    self.fail('incorrect `file_type` in examples.')


if __name__ == '__main__':
    unittest.main()
