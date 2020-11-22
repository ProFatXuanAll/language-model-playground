r"""Dataset module.

All dataset must be import from this file.

Usage:
    import lmp.dataset

    language_model_dataset = lmp.dataset.LanguageModelDataset(...)
    analogy_dataset = lmp.dataset.AnalogyDataset(...)
"""


from lmp.dataset._analogy_dataset import AnalogyDataset
from lmp.dataset._language_model_dataset import LanguageModelDataset
