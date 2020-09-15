r"""Test `lmp.util._dataset._preprocess_word_test_v1`.

Usage:
<<<<<<< HEAD
    python -m unittest test.lmp.util._dataset.test_preprocess_word_test_v1
=======
    python -m unittest test.lmp.util._dataset._preprocess_word_test_v1
>>>>>>> 6bca79baceacf85c5c3683bbfdf586a00484ed19
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

# self-made modules

import lmp.util


# pylint: disable=W0212
class TestPreprocessWordTestV1(unittest.TestCase):
    r"""Test case of `lmp.util._dataset._preprocess_word_test_v1`"""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'
        self.assertEqual(
            inspect.signature(lmp.util._dataset._preprocess_word_test_v1),
            inspect.Signature(
                return_annotation=lmp.dataset.AnalogyDataset
            ),
            msg=msg
        )

    def test_return_type(self):
        r"""Return `lmp.dataset.AnalogyDataset`"""
        msg = 'Must return `lmp.dataset.AnalogyDataset`.'

        dataset = lmp.util._dataset._preprocess_word_test_v1()
        self.assertIsInstance(
            dataset,
            lmp.dataset.AnalogyDataset,
            msg=msg
        )
# pylint: enable=W0212


if __name__ == '__main__':
    unittest.main()
