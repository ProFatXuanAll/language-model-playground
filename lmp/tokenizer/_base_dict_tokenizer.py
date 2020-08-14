r"""Tokenizer base class using `dict` structure.

Usage:
    class CustomTokenizer(BaseDictTokenizer):
        ...
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import json
import os

from typing import Iterable
from typing import List

# 3rd-party modules

from tqdm import tqdm

# self-made modules

import lmp.path

from lmp.tokenizer._base_tokenizer import BaseTokenizer


class BaseDictTokenizer(BaseTokenizer):
    r"""Tokenizer base class using `dict` structure.

    Design philosophy:
        Using `dict` structure to perform token ids lookup is slower compare to
        `list` because python use hash algorithm to implement `dict` and bucket
        size may need to dynamically adjust. We use another `dict` to perform
        inverse lookup, so in theory both `encode` and `decode` have exact same
        speed. But this means `BaseDictTokenizer` will consume much higher
        memory compare to `BaseListTokenizer` implementation.

    Attributes:
        bos_token:
            Token represent the begining of a sequence. Sequences will be
            encoded into following format:
                [bos] t1 t2 ... tn [eos] [pad] [pad] ... [pad]
        eos_token:
            Token represent the end of a sequence. Sequences will be encoded
            into following format:
                [bos] t1 t2 ... tn [eos] [pad] [pad] ... [pad]
        id_to_token:
            Token to id inverse look up data structure. Implemented with `dict`
            data structure.
        is_uncased:
            Whether to differentiate upper cases and lower cases.
        pad_token:
            Token represent padding of a sequence. Only used when sequence
            length is shorter than must.
        token_to_id:
            Token to id look up data structure. Implemented with `dict` data
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

    def reset_vocab(self) -> None:
        r"""Reset vocabulary to initial state.

        Using `dict` structure to implement token look up.
        """
        # Declare vocabulary data structure with `dict`. `token_to_id` serves
        # as token's id look up and `id_to_token` serves as inverse look up.
        self.token_to_id = {}
        self.id_to_token = {}

        # Initialize with special tokens mapping.
        for token_id, token in enumerate(self.__class__.special_tokens()):
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    @classmethod
    def load(cls, experiment: str):
        r"""Load tokenizer JSON file.

        Args:
            experiment:
                Name of the existing experiment.

        Raises:
            FileNotFoundError:
                If directory `experiment` or file `experiment/tokenizer.json`
                does not exist.
            JSONDecodeError:
                If tokenizer is not in JSON format.
            TypeError:
                When `experiment` is not an instance of `str`.
            ValueError:
                When `experiment` is empty string.
        """
        # Type check.
        if not isinstance(experiment, str):
            raise TypeError('`experiment` must be an instance of `str`.')

        # Value check.
        if not experiment:
            raise ValueError('`experiment` must not be empty.')

        file_path = os.path.join(
            lmp.path.DATA_PATH,
            experiment,
            'tokenizer.json'
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} does not exist.')

        with open(file_path, 'r', encoding='utf-8') as input_file:
            obj = json.load(input_file)

        self = cls(is_uncased=obj['is_uncased'])
        self.token_to_id = obj['token_to_id']
        self.id_to_token = {
            v: i for i, v in self.token_to_id.items()
        }

        return self

    @abc.abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        r"""Perform tokenization on input sequence.

        Args:
            sequence:
                Input sequence to be tokenized.

        Raises:
            TypeError:
                When `sequence` is not an instance of `str`.

        Returns:
            Tokens represent input sequence.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `tokenize` not implemented yet.'
        )

    @abc.abstractmethod
    def detokenize(self, tokens: Iterable[str]) -> str:
        r"""Convert tokens back to sequence.

        All subclasses must implement this instance method.

        Args:
            tokens:
                Tokens to be converted.

        Raises:
            TypeError:
                When `tokens` is not an instance of `Iterable[str]`.

        Returns:
            Sequence converted from input tokens.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `detokenize` not implemented yet.'
        )

    def convert_token_to_id(self, token: str) -> int:
        r"""Perform token id look up.

        Args:
            token:
                Look up input token.

        Raises:
            TypeError:
                When `token` is not an instance of `str`.

        Returns:
            Token's id look up result. If `token` does not exist in tokenizer's
            vocabulary, then return unknown word token's id.
        """
        if not isinstance(token, str):
            raise TypeError('`token` must be an instance of `str`.')

        try:
            return self.token_to_id[token]
        except KeyError:
            return self.token_to_id[self.__class__.unk_token]

    def convert_id_to_token(self, token_id: int) -> str:
        r"""Perform token id inverse look up.

        Args:
            token_id:
                Inverse look up token's id.

        Raises:
            TypeError:
                When `token_id` is not an instance of `int`.

        Returns:
            Token id's inverse lookup result. If `token_id` does not exist in
            tokenizer's vocabulary, then return unknown word token.
        """
        if not isinstance(token_id, int):
            raise TypeError('`token_id` must be an instance of `int`.')

        try:
            return self.id_to_token[token_id]
        except KeyError:
            return self.__class__.unk_token

    def build_vocab(
            self,
            batch_sequences: Iterable[str],
            min_count: int = 1
    ) -> None:
        """Build vocabulary for tokenizer.

        Vocabulary is sorted by token's frenquency in descending order.

        Args:
            batch_sequences:
                Vocabulary source.
            min_count:
                Minimum of token's frequency. If token's frequency is smaller
                than `min_count`, then discard that token.

        Raises:
            TypeError:
                When `batch_sequences` is not an instance of `Iterable[str]` or
                `min_count` is not an instance of `int`.
        """
        # Type check.
        if not isinstance(batch_sequences, Iterable):
            raise TypeError(
                '`batch_sequences` must be an instance of `Iterable[str]`.'
            )

        if not isinstance(min_count, int):
            raise TypeError('`min_count` must be an instance of `int`.')

        try:
            token_freq_counter = {}

            for sequence in batch_sequences:
                for token in self.tokenize(sequence):
                    if token not in token_freq_counter:
                        token_freq_counter[token] = 0
                    token_freq_counter[token] += 1
        except TypeError:
            raise TypeError(
                '`batch_sequences` must be an instance of `Iterable[str]`.'
            )

        # Sort tokens based on frequency.
        new_tokens = sorted(
            filter(
                lambda token: (
                    # Filter out tokens having frequency smaller than
                    # `min_count`.
                    token_freq_counter[token] >= min_count and
                    # Filter out tokens already in vocabulary.
                    token not in self.token_to_id
                ),
                token_freq_counter.keys()
            ),
            key=lambda token: token_freq_counter[token],
            reverse=True
        )

        build_vocab_iterator = tqdm(
            new_tokens,
            desc='Build tokneizer vocabulary'
        )

        # New token id must begin with last token id.
        start_token_id = self.vocab_size

        # Add new tokens to vocabulary.
        for fake_token_id, new_token in enumerate(build_vocab_iterator):
            new_token_id = fake_token_id + start_token_id
            self.token_to_id[new_token] = new_token_id
            self.id_to_token[new_token_id] = new_token
