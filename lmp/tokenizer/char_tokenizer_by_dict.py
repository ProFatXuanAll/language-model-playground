
from .base_tokenizer_by_dict import BaseTokenizerByDict
from typing import List

class CharTokenizerByDict(BaseTokenizerByDict):
    r"""Tokenizing sentence by spliting all characters.
    """

    def __init__(self, **kwargs):
        super(CharTokenizerByDict, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return list(sentence)

    def detokenize(self, tokens: List) -> str:
        return ''.join(tokens)
