r"""Test `lmp.util`.

Usage:
    python -m unittest test.lmp.util.__init__
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unittest


class TestUtil(unittest.TestCase):
    r"""Test case for `lmp.util`."""

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent module signature.'

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.util
            # pylint: enable=C0415

            # pylint: disable=W0212
            self.assertTrue(
                inspect.ismodule(lmp.util),
                msg=msg
            )
            # pylint: enable=W0212
        except ImportError:
            self.fail(msg=msg)

    def test_module_attributes(self):
        r"""Declare required module attributes."""
        msg1 = 'Missing module attribute `{}`.'
        msg2 = 'Module attribute `{}` must be a function.'
        msg3 = 'Inconsistent module signature.'
        examples = (
            'load_config',
            'load_dataset',
            'load_dataset_by_config',
            'perplexity_eval',
            'batch_perplexity_eval',
            'generate_sequence',
            'generate_sequence_by_config',
            'load_model',
            'load_model_by_config',
            'load_optimizer',
            'load_optimizer_by_config',
            'set_seed',
            'set_seed_by_config',
            'load_tokenizer',
            'load_tokenizer_by_config',
            'train_model',
            'train_model_by_config',
            'train_tokenizer',
            'train_tokenizer_by_config',
        )

        try:
            # pylint: disable=C0415
            import lmp
            import lmp.util
            # pylint: enable=C0415

            for attr in examples:
                self.assertTrue(
                    hasattr(lmp.util, attr),
                    msg=msg1.format(attr)
                )
                self.assertTrue(
                    inspect.isfunction(getattr(lmp.util, attr)),
                    msg=msg2.format(attr)
                )
        except ImportError:
            self.fail(msg=msg3)


if __name__ == '__main__':
    unittest.main()
