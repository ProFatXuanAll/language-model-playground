r"""Test `lmp.util.set_seed`.

Usage:
    python -m unittest test.lmp.util._seed.test_set_seed
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import math
import random
import unittest

# 3rd-party modules

import numpy as np
import torch

# self-made modules

import lmp.util


class TestSetSeed(unittest.TestCase):
    r"""Test case for `lmp.util.set_seed`."""

    @classmethod
    def setUpClass(cls):
        r"""Setup dynamic seeds."""
        cls.seed_range = list(range(1, 10))

    @classmethod
    def tearDownClass(cls):
        r"""Delete dynamic seeds."""
        del cls.seed_range
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.set_seed),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='seed',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_invalid_input_seed(self):
        r"""Raise exception when input `seed` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `seed` is '
            'invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, -1, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf,
            0j, 1j, '', b'', (), [], {}, set(), object(), lambda x: x, type,
            None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.set_seed(seed=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`seed` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`seed` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_control_random(self):
        r"""Control randomness."""
        msg = 'Must control randomness.'

        for seed in self.__class__.seed_range:
            lmp.util.set_seed(seed)
            r1 = random.randint(0, 10000000)

            lmp.util.set_seed(seed)
            r2 = random.randint(0, 10000000)

            self.assertEqual(r1, r2, msg=msg)

            lmp.util.set_seed(seed)
            n1 = np.random.rand()

            lmp.util.set_seed(seed)
            n2 = np.random.rand()

            self.assertEqual(n1, n2, msg=msg)

            lmp.util.set_seed(seed)
            l1 = torch.nn.Linear(2, 2)

            lmp.util.set_seed(seed)
            l2 = torch.nn.Linear(2, 2)

            for p1, p2 in zip(l1.parameters(), l2.parameters()):
                self.assertTrue((p1 == p2).all().item(), msg=msg)


if __name__ == '__main__':
    unittest.main()
