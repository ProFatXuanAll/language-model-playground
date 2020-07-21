from .base_tokenizer_by_list import BaseTokenizerByList

from typing import List
import re

class WhiteSpaceTokenizerByList(BaseTokenizerByList):
    r"""Tokenizing sentence by spliting spaces.
    """

    def __init__(self, **kwargs):
        super(WhiteSpaceTokenizerByList, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return re.split(r'\s+', sentence)

    def detokenize(self, tokens: List) -> str:
        return ' '.join(tokens)

