r"""Test `lmp.util.set_seed_by_config`.

Usage:
    python -m unittest test.lmp.util._seed.test_set_seed_by_config
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

from itertools import product

# 3rd-party modules

import numpy as np
import torch

# self-made modules

import lmp.config
import lmp.util


class TestSetSeedByConfig(unittest.TestCase):
    r"""Test case for `lmp.util.set_seed_by_config`."""

    @classmethod
    def setUpClass(cls):
        r"""Setup dynamic seeds."""
        cls.config_parameters = {
            'dataset': ['I-AM-TEST-DATASET'],
            'experiment': ['I-AM-TEST-EXPERIMENT'],
            'seed': list(range(1, 10)),
        }

    @classmethod
    def tearDownClass(cls):
        r"""Delete dynamic seeds."""
        del cls.config_parameters
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.set_seed_by_config),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='config',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.config.BaseConfig,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=None
            ),
            msg=msg
        )

    def test_invalid_input_config(self):
        r"""Raise `TypeError` when input `config` is invalid."""
        msg1 = 'Must raise `TypeError` when input `config` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.set_seed_by_config(config=invalid_input)

            self.assertEqual(
                ctx_man.exception.args[0],
                '`config` must be an instance of `lmp.config.BaseConfig`.',
                msg=msg2
            )

    def test_control_random(self):
        r"""Control randomness."""
        msg = 'Must control randomness.'

        for (
                dataset,
                experiment,
                seed
        ) in product(*self.__class__.config_parameters.values()):
            config = lmp.config.BaseConfig(
                dataset=dataset,
                experiment=experiment,
                seed=seed
            )

            lmp.util.set_seed_by_config(config)
            r1 = random.randint(0, 10000000)

            lmp.util.set_seed_by_config(config)
            r2 = random.randint(0, 10000000)

            self.assertEqual(r1, r2, msg=msg)

            lmp.util.set_seed_by_config(config)
            n1 = np.random.rand()

            lmp.util.set_seed_by_config(config)
            n2 = np.random.rand()

            self.assertEqual(n1, n2, msg=msg)

            lmp.util.set_seed_by_config(config)
            l1 = torch.nn.Linear(2, 2)

            lmp.util.set_seed_by_config(config)
            l2 = torch.nn.Linear(2, 2)

            for p1, p2 in zip(l1.parameters(), l2.parameters()):
                self.assertTrue((p1 == p2).all().item(), msg=msg)


if __name__ == '__main__':
    unittest.main()
