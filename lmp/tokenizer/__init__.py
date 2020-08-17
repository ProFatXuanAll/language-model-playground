r"""Tokenizer module.

All tokenizer must import from this file.

Usage:
    import lmp.tokenizer

    tokenizer = lmp.tokenizer.CharDictTokenizer(...)
    tokenizer = lmp.tokenizer.CharListTokenizer(...)
    tokenizer = lmp.tokenizer.WhitespaceDictTokenizer(...)
    tokenizer = lmp.tokenizer.WhitespaceListTokenizer(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

from lmp.tokenizer._base_tokenizer import BaseTokenizer
from lmp.tokenizer._base_dict_tokenizer import BaseDictTokenizer
from lmp.tokenizer._base_list_tokenizer import BaseListTokenizer
from lmp.tokenizer._char_dict_tokenizer import CharDictTokenizer
from lmp.tokenizer._char_list_tokenizer import CharListTokenizer
from lmp.tokenizer._whitespace_dict_tokenizer import WhitespaceDictTokenizer
from lmp.tokenizer._whitespace_list_tokenizer import WhitespaceListTokenizer
