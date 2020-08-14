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
import re

# 3rd-party modules

import pandas as pd
import torch.utils.data

# self-made modules

import lmp.config
import lmp.dataset
import lmp.path


def _preprocess_news_collection(
        column: str) -> lmp.dataset.LanguageModelDataset:
    r"""Preprocessing of news collection dataset and convert to 
    LanguageModelDataset.
    
    Args:
        column:
            Select the part of the data which want to use.

    Returns:
        lmp.dataset.LanguageModelDataset
    """
    file_path = f'{lmp.path.DATA_PATH}/news_collection.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'file {file_path} does not exist.'
        )
    df = pd.read_csv(file_path)
    batch_sequences = df[column].to_list()
    return lmp.dataset.LanguageModelDataset(batch_sequences)


def _preprocess_wiki_tokens(dataset: str) -> lmp.dataset.LanguageModelDataset:
    r"""Preprocess of wiki dataset and convert to LanguageModelDataset.
    
    Args:
        column:
            Select the data which want to use.

    Returns:
        lmp.dataset.LanguageModelDataset
    """
    file_path = f'{lmp.path.DATA_PATH}/wiki.train.tokens'
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'file {file_path} does not exist.'
        )
    with open(file_path, 'r', encoding='utf8') as f:
        df = f.read()

    batch_sequences = list(filter(None, re.split(' =', df.replace('\n', ' '))))
    return lmp.dataset.LanguageModelDataset(batch_sequences)


def _preprocess_word_test_v1_tokens(
) -> lmp.dataset.AnalogyDataset:
    r"""Preprocess word_test_v1 dataset and convert to LanguageModelDataset.
    
    Returns:
        lmp.dataset.AnalogyDataset
    """
    file_path = f'{lmp.path.DATA_PATH}/word-test.v1.txt'
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
) -> torch.utils.data.Dataset:
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
        `lmp.dataset.LanguageModelDataset` instance where samples are sequences.
    """
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
) -> torch.utils.data.Dataset:
    r"""Load dataset from downloaded files.

    Args:
        config:
            Configuration object with attribute `dataset`.

    Returns:
        Same as `load_dataset`.
    """
    return load_dataset(dataset=config.dataset)
