
from .base_tokenizer_by_list import BaseTokenizerByList
from typing import List

# import tokenizer2.base_tokenizer_by_list
class CharTokenizerByList(BaseTokenizerByList):
    r"""Tokenizing sentence by spliting all characters.
    """

    def __init__(self, **kwargs):
        super(CharTokenizerByList, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return list(sentence)

    def detokenize(self, tokens: List) -> str:
        return ''.join(tokens)

