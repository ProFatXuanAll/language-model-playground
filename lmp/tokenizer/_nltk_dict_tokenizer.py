r"""Whitespace Tokenizer using `dict` structure.

Usage:
    from lmp.tokenizer import NltkDictTokenizer

    batch_sequences = (
        'I like apple.',
        'I really like to eat apple.'
    )

    tokenizer = NltkDictTokenizer()
    tokenizer.build_vocab(batch_sequences)

    sequence = batch_sequences[0]

    tokens = tokenizer.tokenize(sequence)
    sequence = tokenizer.detokenize(tokens)

    batch_token_ids = tokenizer.encode(batch_seqeunces)
    batch_sequences = tokenizer.decode(batch_token_ids)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import unicodedata

from typing import List

# 3rd-party modules

import nltk

# self-made modules

from lmp.tokenizer._base_dict_tokenizer import BaseDictTokenizer

class NltkDictTokenizer(BaseDictTokenizer):
    r"""nltk tokenizer using `dict` structure.

    TODO: write perf for speed and memory test.

    Attributes:
        bos_token:
            Token represent the begining of a sequence.
            Sequences will be encoded into following format:
            [BOS] t1 t2 ... tn [EOS].
        eos_token:
            Token represent the end of a sequence.
            Sequences will be encoded into following format:
            [BOS] t1 t2 ... tn [EOS].
        id_to_token:
            Token to id inverse look up data structure.
            Implemented with `dict` data structure.
        is_uncased:
            Whether to differentiate upper cases and lower cases.
        pad_token:
            Padding token.
            Only used when sequence length is shorter than must.
        token_to_id:
            Token to id look up data structure.
            Implemented with `dict` data structure.
        unk_token:
            Token represent unknown words.
            If tokens are not in tokenizer's vocabulary, then tokens will be
            replaced by unknown token.
        vocab_size:
            Vocabulary size of tokenizer.
    """

    def __init__(self, is_uncased: bool = False):
        super().__init__(is_uncased=is_uncased)

    def tokenize(self,sequence: str) -> List[str]:
        r"""Perform tokenization on input sequence.

        Input sequence will first be normalized using unicode's NFKC format.
        If `self.is_uncased == True`, then convert input sequence to lower
        cases.We use nltk.wordpunct_tokenize to split sequence and use reguler
        expression to Eliminate punctuation.

        Args:
            sequence:
                Input sequence to be tokenized.

        Returns:
            Tokens (characters) represent input sequence.
        """

        # NFKC normalization.
        sequence = unicodedata.normalize('NFKC', sequence)

        #Convert into lower cases.
        if self.is_uncased:
            sequence = sequence.lower()

        # Stripping both leading and trailing whitespace characters.
        sequence = sequence.strip()

        #Perform tokenization
        tokens=nltk.wordpunct_tokenize(sequence)
        for i in range(len(tokens)):
            tokens[i]=re.sub('\W+','',tokens[i])
        return tokens

    def detokenize(self, tokens: List) -> str:
        r"""Convert tokens back to sequence.

        Args:
            tokens:
                Tokens to be converted.

        Returns:
            Sequence converted from input tokens.
        """
        return ' '.join(tokens)
