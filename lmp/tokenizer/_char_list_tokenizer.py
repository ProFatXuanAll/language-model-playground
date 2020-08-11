r"""Character tokenizer using `list` structure.

Usage:
    from lmp.tokenizer import CharListTokenizer

    batch_sequences = (
        'I like apple.',
        'I really like to eat apple.'
    )

    tokenizer = CharListTokenizer()
    tokenizer.build_vocab(batch_sequences)

    sequence = batch_sequences[0]

    tokens = tokenizer.tokenize(sequence)
    sequence = tokenizer.detokenize(tokens)

    token_ids = tokenizer.encode(seqeunce)
    sequence = tokenizer.decode(token_ids)

    batch_token_ids = tokenizer.batch_encode(batch_seqeunces)
    batch_sequences = tokenizer.batch_decode(batch_token_ids)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import unicodedata

from typing import Iterable
from typing import List

# self-made modules

from lmp.tokenizer._base_list_tokenizer import BaseListTokenizer


class CharListTokenizer(BaseListTokenizer):
    r"""Character tokenizer using `list` structure.

    Attributes:

        bos_token:
            Token represent the begining of a sequence. Sequences will be
            encoded into following format:
                [BOS] t1 t2 ... tn [EOS] [PAD] [PAD] ... [PAD]
        eos_token:
            Token represent the end of a sequence. Sequences will be encoded
            into following format:
                [BOS] t1 t2 ... tn [EOS] [PAD] [PAD] ... [PAD]
        is_uncased:
            Whether to differentiate upper cases and lower cases.
        pad_token:
            Token represent padding of a sequence. Only used when sequence
            length is shorter than must.
        token_to_id:
            Token to id look up data structure. Implemented with `list` data
            structure.
        unk_token:
            Token represent unknown word in a sequence. If a token is not in
            tokenizer's vocabulary, then that token will be replaced by unknown
            token.
        vocab_size:
            Number of words in tokenizer's vocabulary.

    Raises:
        TypeError:
            When `is_uncased` is not an instance of `bool`.
    """

    def tokenize(self, sequence: str) -> List[str]:
        r"""Perform tokenization on input sequence.

        Input sequence will first be normalized using unicode's NFKC format.
        If `self.is_uncased == True`, then convert input sequence to lower
        cases. We define whitespace characters using python's `re` module with
        pattern `r'\s'` (regular expression for all whitespace characters) for
        the rest of the context. Then we stripped both leading and trailing
        whitespace characters and convert all consecutive whitespace characters
        into single whitespace character. Finally we split sequence into
        characters by passing `str` to `list`.

        Args:
            sequence:
                Input sequence to be tokenized.

        Raises:
            TypeError:
                When `sequence` is not an instance of `str`.

        Returns:
            Tokens (characters) represent input sequence.
        """
        # Type check.
        if not isinstance(sequence, str):
            raise TypeError('`sequence` must be an instance of `str`.')

        # NFKC normalization.
        sequence = unicodedata.normalize('NFKC', sequence)

        # Convert into lower cases.
        if self.is_uncased:
            sequence = sequence.lower()

        # Stripping both leading and trailing whitespace characters.
        sequence = sequence.strip()

        # Convert consecutive whitespace characters into single whitespace
        # character.
        sequence = re.sub(r'\s+', ' ', sequence)

        # Perform tokenization.
        return list(sequence)

    def detokenize(self, tokens: Iterable[str]) -> str:
        r"""Convert tokens back to sequence.

        Since each tokens are originally tokenized as characters, we can simply
        join them into single sequence.

        Args:
            tokens:
                Tokens to be converted.

        Raises:
            TypeError:
                When `tokens` is not an instance of `Iterable[str]`.

        Returns:
            Sequence converted from input tokens.
        """
        # Type check.
        if not isinstance(tokens, Iterable):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        tokens = list(tokens)

        if any(map(lambda token: not isinstance(token, str), tokens)):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        # Perform detokenization.
        return ''.join(tokens)
