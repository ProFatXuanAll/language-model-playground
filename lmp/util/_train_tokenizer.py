r"""Helper function for training tokenizer.

Usage:
    import lmp

    lmp.util.train_tokenizer(...)
    lmp.util.train_tokenizer_by_config(...)
"""
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

import lmp.config
import lmp.dataset
import lmp.tokenizer


def train_tokenizer(
        dataset: lmp.dataset.LanguageModelDataset,
        min_count: int,
        tokenizer: lmp.tokenizer.BaseTokenizer
):
    r"""Helper function for training tokenizer.

    Args:
        dataset:
            List of sequences.
        min_count:
            Minimum frequency required for each token.
        tokenizer:
            Training tokenizer instance.
    """
    tokenizer.build_vocab(
        batch_sequences=dataset,
        min_count=min_count
    )


def train_tokenizer_by_config(
        config: lmp.config.BaseConfig,
        dataset: lmp.dataset.LanguageModelDataset,
        tokenizer: lmp.tokenizer.BaseTokenizer
):
    r"""Helper function for training tokenizer.

    Args:
        config:
            Configuration object with attribute `min_count`.
        dataset:
            List of sequences.
        tokenizer:
            Training tokenizer instance.
    """
    train_tokenizer(
        dataset=dataset,
        min_count=config.min_count,
        tokenizer=tokenizer
    )
