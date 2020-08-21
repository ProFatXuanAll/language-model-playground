r"""Helper function for loading dataset.

Usage:
    import lmp.util

    dataset = lmp.util.load_dataset(...)
    dataset = lmp.util.load_dataset_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# 3rd-party modules

import pandas as pd

# self-made modules

import lmp.config
import lmp.dataset
import lmp.path


def load_dataset(dataset: str) -> lmp.dataset.BaseDataset:
    r"""Load dataset from downloaded files.

    Args:
        dataset:
            Name of the dataset to perform experiment.

    Raises:
        TypeError:
            When `dataset` is not an instance of `str`.
        ValueError:
            If `dataset` does not support.
        FileNotFoundError
            If `dataset` does not exist.

    Returns:
        `lmp.dataset.BaseDataset` instance where samples are sequences.
    """
    if not isinstance(dataset, str):
        raise TypeError('`dataset` must be instance of `str`.')

    if dataset == 'news_collection_desc':
        file_path = f'{lmp.path.DATA_PATH}/news_collection.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'file {file_path} does not exist.'
            )
        df = pd.read_csv(file_path)
        df = df.dropna()
        batch_sequences = df['desc'].to_list()
        return lmp.dataset.BaseDataset(batch_sequences)

    if dataset == 'news_collection_title':
        file_path = f'{lmp.path.DATA_PATH}/news_collection.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'file {file_path} does not exist.'
            )
        df = pd.read_csv(file_path)
        batch_sequences = df['title'].to_list()
        return lmp.dataset.BaseDataset(batch_sequences)

    raise ValueError(
        f'dataset `{dataset}` does not support.\nSupported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--dataset {option}',
            [
                'news_collection_desc',
                'news_collection_title',
            ]
        )))
    )


def load_dataset_by_config(
        config: lmp.config.BaseConfig
) -> lmp.dataset.BaseDataset:
    r"""Load dataset from downloaded files.

    Args:
        config:
            Configuration object with attribute `dataset`.

    Raise:
        TypeError:
            When `config` is not an instance of `lmp.config.BaseConfig`.

    Returns:
        Same as `load_dataset`.
    """
    if not isinstance(config, lmp.config.BaseConfig):
        raise TypeError(
            '`config` must be an instance of `lmp.config.BaseConfig`.'
        )

    return load_dataset(dataset=config.dataset)
