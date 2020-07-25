r"""Helper function for loading tokenizer.

Usage:
    import lmp

    tokenizer = lmp.util.load_tokenizer(...)
    tokenizer = lmp.util.load_tokenizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

import lmp.tokenizer
import lmp.config


def load_tokenizer(
        checkpoint: int,
        experiment: str,
        is_uncased: bool = False,
        tokenizer_class: str = 'char-list'
) -> lmp.tokenizer.BaseTokenizer:
    r"""Helper function for constructing tokenizer.

    Load pre-trained tokenizer when `checkpoint != -1`.

    Args:
        checkpoint:
            Whether to load pre-trained tokenizer.
        experiment:
            Name of the pre-trained experiment.
        is_uncased:
            Whether to convert upper cases to lower cases.
        tokenizer_class:
            Which tokenizer class to construct.

    Raises:
        ValueError:
            If `tokenizer_class` does not support.

    Returns:
        `CharDictTokenizer` if `tokenizer_class == 'char_dict'`.
        `CharListTokenizer` if `tokenizer_class == 'char_list'`.
        `WhitespaceDictTokenizer` if `tokenizer_class == 'whitespace_dict'`.
        `WhitespaceListTokenizer` if `tokenizer_class == 'whitespace_list'`.
    """
    if tokenizer_class == 'char_dict':
        tokenizer = lmp.tokenizer.CharDictTokenizer(is_uncased=is_uncased)
    elif tokenizer_class == 'char_list':
        tokenizer = lmp.tokenizer.CharListTokenizer(is_uncased=is_uncased)
    elif tokenizer_class == 'whitespace_dict':
        tokenizer = lmp.tokenizer.WhitespaceDictTokenizer(
            is_uncased=is_uncased)
    elif tokenizer_class == 'whitespace_list':
        tokenizer = lmp.tokenizer.WhitespaceListTokenizer(
            is_uncased=is_uncased)
    else:
        raise ValueError(
            f'`{tokenizer_class}` does not support.'
        )

    if checkpoint != -1:
        tokenizer = tokenizer.load(experiment=experiment)

    return tokenizer


def load_tokenizer_by_config(
        checkpoint: int,
        config: lmp.config.BaseConfig
) -> lmp.tokenizer.BaseTokenizer:
    r"""Helper function for constructing tokenizer.

    Load pre-trained tokenizer when `checkpoint != -1`.

    Args:
        checkpoint:
            Whether to load pre-trained tokenizer.
        config:
            Configuration object with attributes `is_uncased`,
            `experiment` and `tokenizer_class`.

    Returns:
        Same as `load_tokenizer`.
    """
    return load_tokenizer(
        checkpoint=checkpoint,
        experiment=config.experiment,
        is_uncased=config.is_uncased,
        tokenizer_class=config.tokenizer_class
    )
