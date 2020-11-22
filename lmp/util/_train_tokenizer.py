r"""Helper function for training tokenizer.

Usage:
    import lmp.util

    lmp.util.train_tokenizer(...)
    lmp.util.train_tokenizer_by_config(...)
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



import lmp.config
import lmp.dataset
import lmp.tknzr


def train_tokenizer(
        dataset: lmp.dataset.LanguageModelDataset,
        min_count: int,
        tokenizer: lmp.tknzr.BaseTknzr
) -> None:
    r"""Helper function for training tokenizer.

    Args:
        dataset:
            Source of text samples to train on.
        min_count:
            Minimum frequency required for each token.
        tokenizer:
            Training tokenizer instance.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.
        ValueError:
            When `min_count` is smaller than `1`.
    """
    # Type check.
    if not isinstance(dataset, lmp.dataset.LanguageModelDataset):
        raise TypeError(
            '`dataset` must be an instance of `lmp.dataset.LanguageModelDataset`.'
        )

    if not isinstance(min_count, int):
        raise TypeError('`min_count` must be an instance of `int`.')

    if not isinstance(tokenizer, lmp.tknzr.BaseTknzr):
        raise TypeError(
            '`tokenizer` must be an instance of `lmp.tknzr.BaseTknzr`.'
        )

    # Value check.
    if min_count < 1:
        raise ValueError('`min_count` must be bigger than or equal to `1`.')

    tokenizer.build_vocab(batch_sequences=dataset, min_count=min_count)


def train_tokenizer_by_config(
        config: lmp.config.BaseConfig,
        dataset: lmp.dataset.LanguageModelDataset,
        tokenizer: lmp.tknzr.BaseTknzr
) -> None:
    r"""Helper function for training tokenizer.

    Args:
        config:
            Configuration object with attribute `min_count`.
        dataset:
            Source of text samples to train on.
        tokenizer:
            Training tokenizer instance.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.
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
