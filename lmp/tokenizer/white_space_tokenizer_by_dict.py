from .base_tokenizer_by_dict import BaseTokenizerByDict

from typing import List
import re

class WhiteSpaceTokenizerByDict(BaseTokenizerByDict):
    r"""Tokenizing sentence by spliting spaces.
    """

    def __init__(self, **kwargs):
        super(BaseTokenizerByDict, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return re.split(r'\s+', sentence)

    def detokenize(self, tokens: List) -> str:
        return ' '.join(tokens)
