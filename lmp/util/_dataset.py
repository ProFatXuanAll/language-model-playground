r"""Helper function for loading dataset.

Usage:
    import lmp

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


def load_dataset(
        dataset: str
) -> lmp.dataset.BaseDataset:
    r"""Load dataset from downloaded files.

    Args:
        dataset:
            Name of the dataset to perform experiment.

    Raises:
        ValueError:
            If `dataset` does not support.
        FileNotFoundError
            If `dataset` does not exist.

    Returns:
        `lmp.dataset.BaseDataset` instance where samples are sequences.
    """
    if dataset == 'news_collection':
        file_path = f'{lmp.path.DATA_PATH}/news_collection.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'file {file_path} does not exist.'
            )
        df = pd.read_csv(file_path)
        batch_sequences = df['title'].to_list()
        return lmp.dataset.BaseDataset(batch_sequences)
    raise ValueError(
        '`dataset` does not support.'
    )


def load_dataset_by_config(
        config: lmp.config.BaseConfig
) -> lmp.dataset.BaseDataset:
    r"""Load dataset from downloaded files.

    Args:
        config:
            Configuration object with attribute `dataset`.

    Returns:
        Same as `load_dataset`.
    """
    return load_dataset(dataset=config.dataset)
