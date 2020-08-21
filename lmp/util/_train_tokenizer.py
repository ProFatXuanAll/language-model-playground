r"""Helper function for training tokenizer.

Usage:
    import lmp.util

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
        dataset: lmp.dataset.BaseDataset,
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

    Raises:
        TypeError:
            When `dataset` is not an instance of `lmp.dataset.BaseDataset`, 
            `min_count` is not an instance of `int` or `tokenizer` is not an
            instance of `lmp.tokenizer.BaseTokenizer`.
        ValueError:
            When `min_count` is smaller than `1`.
    """
    # Type check.
    if not isinstance(dataset, lmp.dataset.BaseDataset):
        raise TypeError(
            '`dataset` must be an instance of `lmp.dataset.BaseDataset`.'
        )

    if not isinstance(min_count, int):
        raise TypeError('`min_count` must be an instance of `int`.')

    if not isinstance(tokenizer, lmp.tokenizer.BaseTokenizer):
        raise TypeError(
            '`tokenizer` must be an instance of `lmp.tokenizer.BaseTokenizer`.'
        )

    # Value check.
    if min_count < 1:
        raise ValueError('`min_count` must be bigger than or equal to `1`.')


    tokenizer.build_vocab(
        batch_sequences=dataset,
        min_count=min_count
    )


def train_tokenizer_by_config(
        config: lmp.config.BaseConfig,
        dataset: lmp.dataset.BaseDataset,
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
    
    Raises:
        TypeError:
            When `config` is not an instance of `lmp.config.BaseConfig`.
    """
    # Type check.
    if not isinstance(config, lmp.config.BaseConfig):
        raise TypeError(
            '`config` must be an instance of `lmp.config.BaseConfig`.'
        )

    train_tokenizer(
        dataset=dataset,
        min_count=config.min_count,
        tokenizer=tokenizer
    )
