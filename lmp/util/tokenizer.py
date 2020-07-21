r"""Helper function for loading tokenizer.

Usage:
    tokenizer = lmp.util.load_saved_tokenizer()
"""

from typing import Union, List

import lmp.tokenizer
import lmp.config


def load_saved_tokenizer(file_path: str, tokenizer_class: str = 'list') -> Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict]:
    r"""Decide to load which saved tokenizer.

    Args:
        file_path:
            Location of tokenizer's pickle file
        tokenizer_class:
            Decide to use which tokenizer, list or dict
            Tokenizer's token_to_id is implemented in different structure(list or dict).
    Returns:
        lmp.tokenizer.CharTokenizerByDict().load_from_file(f'{file_path}/tokenizer.pickle')
        lmp.tokenizer.CharTokenizerByList().load_from_file(f'{file_path}/tokenizer.pickle')
    """
    if tokenizer_class.lower() not in ['list', 'dict']:
        raise ValueError(
            f'`{tokenizer_class}` is not exist, please input list or dict')
    if tokenizer_class.lower() == 'dict':
        return lmp.tokenizer.CharTokenizerByDict.load_from_file(file_path)
    elif tokenizer_class.lower() == 'list':
        return lmp.tokenizer.CharTokenizerByList.load_from_file(file_path)


def load_blank_tokenizer(tokenizer_class: str = 'list') -> Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict]:
    r"""Decide to use which blank tokenizer.

    Args:
        tokenizer_class:
            Decide to use which tokenizer, list or dict
            Tokenizer's token_to_id is implemented in different structure(list or dict).
    Returns:
        lmp.tokenizer.CharTokenizerByDict()
        lmp.tokenizer.CharTokenizerByList()
    """
    if tokenizer_class.lower() not in ['list', 'dict']:
        raise ValueError(
            f'`{tokenizer_class}` is not exist, please input list or dict')
    if tokenizer_class.lower() == 'dict':
        return lmp.tokenizer.CharTokenizerByDict()
    elif tokenizer_class.lower() == 'list':
        return lmp.tokenizer.CharTokenizerByList()


def load_tokenizer_by_config(config: lmp.config.BaseConfig,
                             checkpoint: int,
                             file_path: str,
                             sentneces: List[str] = ['']):

    if checkpoint > 0:
        return load_saved_tokenizer(file_path, config.tokenizer_class)
    else:
        tokenizer = load_blank_tokenizer(config.tokenizer_class)
        tokenizer.build_dict(sentneces, config.min_count, config.is_uncased)

        return tokenizer
