r"""Test `lmp.config.BaseConfig.device`.

Usage:
    python -m unittest test/lmp/config/test_device.py
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest

# 3rd-party modules

import torch

# self-made modules

from lmp.config import BaseConfig


class TestIter(unittest.TestCase):
    r"""Test case for `lmp.config.BaseConfig.device`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent property signature.'

        self.assertTrue(
            inspect.isdatadescriptor(BaseConfig.device),
            msg=msg
        )
        self.assertFalse(
            inspect.isfunction(BaseConfig.device),
            msg=msg
        )
        self.assertFalse(
            inspect.ismethod(BaseConfig.device),
            msg=msg
        )

    def test_expected_return(self):
        r"""Return expected `torch.device`."""
        msg = 'Inconsistent `torch.device`.'
        examples = (torch.device('cpu'), torch.device('cuda'))

        self.assertIn(
            BaseConfig(dataset='test', experiment='test').device,
            examples,
            msg=msg
        )


if __name__ == '__main__':
    unittest.main()
