r"""Helper function for loading tokenizer.

Usage:
    tokenizer = lmp.util.load_tokenizer()
"""

from typing import Union

import lmp.tokenizer


def load_tokenizer(model_path: str, tokenizer_type: str = 'list') -> Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict]:
    r"""Decide to load which saved tokenizer.

    Args:
        model_path:
            Location of tokenizer's pickle file
        tokenizer_type:
            Decide to use which tokenizer, list or dict
            Tokenizer's token_to_id is implemented in different structure(list or dict).
    Returns:
        lmp.tokenizer.CharTokenizerByDict().load_from_file(f'{model_path}/tokenizer.pickle')
        lmp.tokenizer.CharTokenizerByList().load_from_file(f'{model_path}/tokenizer.pickle')
    """
    if tokenizer_type.lower() not in ['list', 'dict']:
        raise ValueError(
            f'`{args.tokenizer}` is not exist, please input list or dict')
    if tokenizer_type.lower() == 'dict':
        return lmp.tokenizer.CharTokenizerByDict.load_from_file(f'{model_path}/tokenizer.pickle')
    elif tokenizer_type.lower() == 'list':
        return lmp.tokenizer.CharTokenizerByList.load_from_file(f'{model_path}/tokenizer.pickle')


def load_blank_tokenizer(tokenizer_type: str = 'list') -> Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict]:
    r"""Decide to use which blank tokenizer.

    Args:
        tokenizer_type:
            Decide to use which tokenizer, list or dict
            Tokenizer's token_to_id is implemented in different structure(list or dict).
    Returns:
        lmp.tokenizer.CharTokenizerByDict()
        lmp.tokenizer.CharTokenizerByList()
    """
    if tokenizer_type.lower() not in ['list', 'dict']:
        raise ValueError(
            f'`{args.tokenizer}` is not exist, please input list or dict')
    if tokenizer_type.lower() == 'dict':
        return lmp.tokenizer.CharTokenizerByDict()
    elif tokenizer_type.lower() == 'list':
        return lmp.tokenizer.CharTokenizerByList()
