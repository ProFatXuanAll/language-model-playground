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
import unittest

from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp


class TestLoadModelSetSeedByConfig(unittest.TestCase):
    r"""Test Case for `lmp.util.set_seed_by_config`."""

    @classmethod
    def setUpClass(cls):
        cls.seed_range = [2, 5, 7, 18]

    @classmethod
    def tearDownClass(cls):
        del cls.seed_range
        gc.collect()

    def setUp(self):
        r"""Set up some random seeds."""
        self.config_obj = []
        self.rand_obj = []
        for seed in self.__class__.seed_range:
            config = lmp.config.BaseConfig(
                dataset='news_collection',
                experiment='util_set_seed_by_config_unittest',
                seed=seed
            )
            self.config_obj.append(config)

            lmp.util.set_seed_by_config(config)
            num = torch.rand(seed)
            self.rand_obj.append(num)

    def tearDown(self):
        r"""Delete parameters for `set_seed`."""
        del self.config_obj
        del self.rand_obj
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
                return_annotation=inspect.Parameter.empty
            ),
            msg=msg
        )

    def test_invalid_input_config(self):
        r"""Raise when `config` is invalid."""
        msg1 = 'Must raise `TypeError` when `config` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            1, 0, -1, True, False, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', [], (), {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.set_seed_by_config(config=invalid_input)

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`config` must be an instance of `lmp.config.BaseConfig`.',
                    msg=msg2
                )

    def test_rand_num(self):
        r"""Test `set_seed_by_config` function normally."""
        msg = 'Inconsistent error message.'
        examples = (
            (
                self.config_obj[i],
                self.rand_obj[i],
            )
            for i in range(len(self.__class__.seed_range))
        )

        for config, ans in examples:
            lmp.util.set_seed_by_config(config)
            num = torch.rand(config.seed)
            for i in range(ans.size(0)):
                self.assertEqual(
                    num[i].item(),
                    ans[i].item(),
                    msg=msg
                )
