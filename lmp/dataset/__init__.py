r"""Dataset module.

All dataset must be import from this file.

Usage:
    import lmp.dataset

    language_model_dataset = lmp.dataset.LanguageModelDataset(...)
    analogy_dataset = lmp.dataset.AnalogyDataset(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

from lmp.dataset._language_model_dataset import LanguageModelDataset
from lmp.dataset._analogy_dataset import AnalogyDataset
