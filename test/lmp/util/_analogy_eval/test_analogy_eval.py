r"""Test `lmp.util.analogy_eval`.

Usage:
    python -m unittest test.lmp.util._analogy_eval.test_analogy_eval
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

# 3rd-party modules

import torch

# self-made modules

import lmp.model.BaseRNNModel
import lmp.util._analogy_eval.analogy_eval
from lmp.dataset._analogy_dataset import AnalogyDataset
from lmp.dataset._language_model_dataset import LanguageModelDataset

class TestAnalogyEval(unittest.TestCase):
    r"""Test case of `lmp.util.analogy_eval"""