r"""Helper function for loading dataset.

Usage:
    import lmp.util

    dataset = lmp.util.load_dataset(...)
    dataset = lmp.util.load_dataset_by_config(...)
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import unicodedata

from typing import Union



import pandas as pd



import lmp.config
import lmp.dataset
import lmp.path


def _preprocess_news_collection(
        column: str
) -> lmp.dataset.LanguageModelDataset:
    r"""Preprocess `news_collection.csv` and convert into `lmp.dataset.LanguageModelDataset`.

    Args:
        column:
            Column name of `news_collection.csv`. Must be either `title` or `desc`.

    Raises:
        FileNotFoundError:
            When file does not exist.
        KeyError:
            When `column` is not available.
        TypeError:
            When `column` is not instance of `str`.

    Returns:
        `lmp.dataset.LanguageModelDataset` from `news_collection.csv`.
    """
    if not isinstance(column, str):
        raise TypeError('`column` must be an instance of `str`.')

    file_path = os.path.join(f'{lmp.path.DATA_PATH}', 'news_collection.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'file {file_path} does not exist.')

    df = pd.read_csv(file_path)
    if column not in df.columns:
        raise KeyError('`column` is not available.')

    data = df[column].dropna().to_list()

    # Normalized by unicode NFKC.
    data = [unicodedata.normalize('NFKC', sample) for sample in data]

    # Convert all new lines and consecutive whitespace into single whitespace.
    data = [re.sub(r'\s+', ' ', sample) for sample in data]

    # Strip leading and trailing whitespaces.
    data = [sample.strip() for sample in data]

    return lmp.dataset.LanguageModelDataset(batch_sequences=data)


def _preprocess_wiki_tokens(split: str) -> lmp.dataset.LanguageModelDataset:
    r"""Preprocess `wiki.*.tokens` and convert into `lmp.dataset.LanguageModelDataset`.

    Args:
        split:
            Split of the Wiki long term dependency language modeling dataset.
            Must be either `train`, `valid` or `test`.

    Raises:
        FileNotFoundError:
            When file does not exist.
        TypeError:
            When `split` is not instance of `str`.

    Returns:
        `lmp.dataset.LanguageModelDataset` from `wiki.*.tokens`.
    """
    if not isinstance(split, str):
        raise TypeError('`split` must be an instance of `str`.')

    file_path = os.path.join(f'{lmp.path.DATA_PATH}', f'wiki.{split}.tokens')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'file {file_path} does not exist.')

    with open(file_path, 'r', encoding='utf8') as input_file:
        data = input_file.read()

    # Split based on section pattern.
    data = re.split(r' \n( =){1,3} .+ (= ){1,3}\n ', data)
    data = list(filter(
        lambda sample: sample.strip()
        and not re.match(r'( =){1,3}', sample)
        and not re.match(r'(= ){1,3}', sample),
        data
    ))

    # Normalized by unicode NFKC.
    data = [unicodedata.normalize('NFKC', sample) for sample in data]

    # Convert all new lines and consecutive whitespace into single whitespace.
    data = [re.sub(r'\s+', ' ', sample) for sample in data]

    # Strip leading and trailing whitespaces.
    data = [sample.strip() for sample in data]

    return lmp.dataset.LanguageModelDataset(batch_sequences=data)


def _preprocess_word_test_v1() -> lmp.dataset.AnalogyDataset:
    r"""Preprocess `word-test.v1.txt` and convert into `lmp.dataset.AnalogyDataset`.

    Returns:
        `lmp.dataset.AnalogyDataset` from `word-test.v1.txt`.
    """
    file_path = os.path.join(f'{lmp.path.DATA_PATH}', 'word-test.v1.txt')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'file {file_path} does not exist.')

    with open(file_path, 'r', encoding='utf8') as input_file:
        # Remove first line since it is just copyright.
        samples = [line.strip() for line in input_file.readlines()][1:]

    # Parsing.
    category = ''
    parsing_samples = []
    for sample in samples:
        # Category line.
        if re.match(r':', sample):
            category = sample[2:]
            continue

        # Word analogy line.
        parsing_samples.append(re.split(r'\s+', sample) + [category])

    return lmp.dataset.AnalogyDataset(samples=parsing_samples)


def load_dataset(
        dataset: str
) -> Union[lmp.dataset.AnalogyDataset, lmp.dataset.LanguageModelDataset]:
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
        return _preprocess_wiki_tokens(split='train')

    if dataset == 'wiki_valid_tokens':
        return _preprocess_wiki_tokens(split='valid')

    if dataset == 'wiki_test_tokens':
        return _preprocess_wiki_tokens(split='test')

    if dataset == 'word_test_v1':
        return _preprocess_word_test_v1()

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
) -> Union[lmp.dataset.AnalogyDataset, lmp.dataset.LanguageModelDataset]:
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
