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
import re

from typing import Union

# 3rd-party modules

import pandas as pd
import torch.utils.data

# self-made modules

import lmp.config
import lmp.dataset
import lmp.path


def _preprocess_news_collection(
        column: str
) -> lmp.dataset.LanguageModelDataset:
    r"""Preprocessing of news collection dataset and convert to
    `lmp.dataset.LanguageModelDataset`.

    Args:
        column:
            Select the part of the data which want to use.

    Returns:
        `lmp.dataset.LanguageModelDataset`
    """
    file_path = os.path.join(f'{lmp.path.DATA_PATH}', 'news_collection.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'file {file_path} does not exist.')

    df = pd.read_csv(file_path).dropna()
    batch_sequences = df[column].to_list()
    return lmp.dataset.LanguageModelDataset(batch_sequences)


def _preprocess_wiki_tokens(dataset: str) -> lmp.dataset.LanguageModelDataset:
    r"""Preprocess of wiki dataset and convert to
    lmp.dataset.LanguageModelDataset`.

    Args:
        column:
            Select the data which want to use.

    Returns:
        `lmp.dataset.LanguageModelDataset`
    """
    file_path = os.path.join(f'{lmp.path.DATA_PATH}', 'wiki.train.tokens')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'file {file_path} does not exist.')

    with open(file_path, 'r', encoding='utf8') as input_file:
        df = input_file.read()

    batch_sequences = list(filter(None, re.split(' =', df.replace('\n', ' '))))
    return lmp.dataset.LanguageModelDataset(batch_sequences)


def _preprocess_word_test_v1_tokens() -> lmp.dataset.AnalogyDataset:
    r"""Preprocess word_test_v1 dataset and convert to
    `lmp.dataset.AnalogyDataset`.

    Returns:
        `lmp.dataset.AnalogyDataset`
    """
    file_path = os.path.join(f'{lmp.path.DATA_PATH}', 'word-test.v1.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'file {file_path} does not exist.'
        )
    with open(file_path, 'r', encoding='utf8') as txt_file:
        samples = [line.strip() for line in txt_file.readlines()]

    # Parsing.
    category = ''
    parsing_samples = []
    for sample in samples:
        if re.match(r'^:', sample):
            category = sample[2:]
            continue

        parsing_samples.append(re.split(r'\s+', sample)[:4] + [category])
    return lmp.dataset.AnalogyDataset(parsing_samples)


def load_dataset(
        dataset: str
) -> Union[lmp.dataset.LanguageModelDataset, lmp.dataset.AnalogyDataset]:
    r"""Load dataset from downloaded files.

    Supported options:
        --dataset news_collection_desc
        --dataset news_collection_title
        --dataset wiki_train_tokens
        --dataset wiki_valid_tokens
        --dataset wiki_test_tokens
        --dataset word_test_v1

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
        `lmp.dataset.LanguageModelDataset` instance where samples are sequences.
        `lmp.dataset.AnalogyDataset` instance where sample is used for analogy test.
    """
    # Type check.
    if not isinstance(dataset, str):
        raise TypeError('`dataset` must be an instance of `str`.')

    if dataset == 'news_collection_desc':
        return _preprocess_news_collection(column='desc')

    if dataset == 'news_collection_title':
        return _preprocess_news_collection(column='title')

    if dataset == 'wiki_train_tokens':
        return _preprocess_wiki_tokens(dataset='train')

    if dataset == 'wiki_valid_tokens':
        return _preprocess_wiki_tokens(dataset='valid')

    if dataset == 'wiki_test_tokens':
        return _preprocess_wiki_tokens(dataset='test')

    if dataset == 'word_test_v1':
        return _preprocess_word_test_v1_tokens()

    raise ValueError(
        f'dataset `{dataset}` does not support.\nSupported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--dataset {option}',
            [
                'news_collection_desc',
                'news_collection_title',
                'wiki_test_tokens',
                'wiki_train_tokens',
                'wiki_valid_tokens',
                'word_test_v1'
            ]
        )))
    )


def load_dataset_by_config(
        config: lmp.config.BaseConfig
) -> Union[lmp.dataset.LanguageModelDataset, lmp.dataset.AnalogyDataset]:
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
    # Type check.
    if not isinstance(config, lmp.config.BaseConfig):
        raise TypeError(
            '`config` must be an instance of `lmp.config.BaseConfig`.'
        )

    return load_dataset(dataset=config.dataset)
