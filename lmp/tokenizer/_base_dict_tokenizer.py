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

from typing import Dict
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

    Raises:
        TypeError:
            When `is_uncased` is not instance of `bool`.
    """

    def reset_vocab(self) -> None:
        r"""Reset vocabulary to initial state.

        Using `dict` structure to implement token look up.
        """
        # Declare vocabulary data structure with `dict`.
        # `token_to_id` serves as token's id look up.
        # and `id_to_token` serves as inverse look up.
        self.token_to_id: Dict = {}
        self.id_to_token: Dict = {}

        # Initialize special tokens mapping.
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
                When `experiment` is not instance of `str`.
            ValueError:
                When `experiment` is empty string.
        """
        # Type check.
        if not isinstance(experiment, str):
            raise TypeError('`experiment` must be instance of `str`.')

        # Value check.
        if not experiment:
            raise ValueError('`experiment` must not be empty.')

        file_path = os.path.join(
            lmp.path.DATA_PATH,
            experiment,
            'tokenizer.json'
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'file {file_path} does not exist.')

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

        All subclasses must implement this instance method and must convert
        sequence cases based on `self.is_uncased`.

        Args:
            sequence:
                Input sequence to be tokenized.

        Raises:
            TypeError:
                When `sequence` is not instance of `str`.

        Returns:
            Tokens represent input sequence.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `tokenize` not implemented yet.'
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
                When `tokens` is not instance of `Iterable[str]`.

        Returns:
            Sequence converted from input tokens.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `detokenize` not implemented yet.'
        )

    def convert_token_to_id(self, token: str) -> int:
        r"""Perform token id look up.

        Args:
            token:
                Look up input token.

        Raises:
            TypeError:
                When `token` is not instance of `str`.

        Returns:
            Token's id look up result. If `token` does not exist in tokenizer's
            vocabulary, then return unknown word token's id.
        """
        if not isinstance(token, str):
            raise TypeError('`token` must be instance of `str`.')

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
                When `token_id` is not instance of `int`.

        Returns:
            Token id's inverse lookup result. If `token_id` does not exist in
            tokenizer's vocabulary, then return unknown word token.
        """
        if not isinstance(token_id, int):
            raise TypeError('`token_id` must be instance of `int`.')

        try:
            return self.id_to_token[token_id]
        except KeyError:
            return self.__class__.unk_token

    def encode(
            self,
            sequence: str,
            max_seq_len: int = -1
    ) -> List[int]:
        r"""Encode sequence into token ids.

        Token ids have following format:
            [BOS] t1 t2 ... tn [EOS] [PAD] ... [PAD]

        Args:
            sequence:
                Sequence to be encoded.
            max_seq_len:
                Whether to truncate or pad sequence to specified length.
                If `max_seq_len == 0`, then sequence will not truncate or pad.
                If `max_seq_len > 0`, then sequence will truncate to
                `max_seq_len` when sequence length is longer than `max_seq_len`
                and pad to `max_seq_len` when sequence length is shorter than
                `max_seq_len`.

        Raises:
            TypeError:
                When `sequence` is not instance of `str` or `max_seq_len` is
                not instance of `int`.

        Returns:
            Token ids encoded from `sequence`.
        """
        # Type check.
        if not isinstance(sequence, str):
            raise TypeError('`sequence` must be instance of `str`.')

        if not isinstance(max_seq_len, int):
            raise TypeError('`max_seq_len` must be instance of `int`.')

        token_ids = self.convert_tokens_to_ids(self.tokenize(sequence))

        # Truncate to max sequence length,
        # -2 for `[BOS]` and `[EOS]`.
        if max_seq_len > 0:
            token_ids = token_ids[:max_seq_len - 2]

        # Encode token_ids with `[BOS]` and `[EOS]`.
        token_ids = [
            self.token_to_id[self.__class__.bos_token],
            *token_ids,
            self.token_to_id[self.__class__.eos_token]
        ]

        # Calculate padding length.
        padding_len = max(0, max_seq_len - len(token_ids))

        # Pad to max sequence length.
        return token_ids + [
            self.token_to_id[self.__class__.pad_token]
            for _ in range(padding_len)
        ]

    def decode(
            self,
            token_ids: Iterable[int],
            remove_special_tokens: bool = False
    ) -> str:
        r"""Decode token ids into sequence.

        Args:
            token_ids:
                Token ids to be decoded.
            remove_special_tokens:
                Whether to remove special tokens.
                If `remove_special_tokens == True`, then remove all special
                tokens except unknown word's token.
                See class docstring for more details on special tokens.

        Raises:
            TypeError:
                When `token_ids` is not instance of `Iterable[int]` or
                `remove_special_tokens` is not instance of `bool`.

        Returns:
            Sequence decoded from `token_ids`.
        """
        # Type check.
        if not isinstance(token_ids, Iterable):
            raise TypeError('`token_ids` must be instance of `Iterable[int]`.')

        token_ids = list(token_ids)

        if not isinstance(remove_special_tokens, bool):
            raise TypeError(
                '`remove_special_tokens` must be instance of `bool`.')

        if remove_special_tokens:
            # Get special tokens' ids except unknown token.
            special_token_ids = list(
                map(
                    lambda token: self.token_to_id[token],
                    filter(
                        lambda token: token != self.__class__.unk_token,
                        self.__class__.special_tokens()
                    )
                )
            )
            # Filter out special tokens' ids
            # and keep unknown token ids if presented.
            token_ids = list(filter(
                lambda token_id: token_id not in special_token_ids,
                token_ids
            ))
        try:
            sequence = self.detokenize(self.convert_ids_to_tokens(token_ids))
        except TypeError:
            raise TypeError('`token_ids` must be instance of `Iterable[int]`.')

        return sequence

    def batch_encode(
            self,
            batch_sequences: Iterable[str],
            max_seq_len: int = -1
    ) -> List[List[int]]:
        r"""Encode batch of sequence into batch of token ids.

        See `encode` for tokens' ids format.

        Args:
            sequence:
                Sequence to be encoded.
            max_seq_len:
                Whether to truncate or pad each sequences to specified length.
                If `max_seq_len == 0`, then each sequences will not be
                truncated but padded to current batch's maximum sequence
                length. If `max_seq_len > 0`, then each sequences will be
                truncated to `max_seq_len` when individual sequence length is
                longer than `max_seq_len` and padded to `max_seq_len` when
                individual sequence length is shorter than `max_seq_len`.

        Raises:
            TypeError:
                When `batch_sequences` is not instance of `Iterable[str]` or
                `max_seq_len` is not instance of `int`.

        Returns:
            Batch of token ids encoded from `batch_sequence`.
        """
        # Type check.
        if not isinstance(batch_sequences, Iterable):
            raise TypeError(
                '`batch_sequences` must be instance of `Iterable[str]`.'
            )
        if not isinstance(max_seq_len, int):
            raise TypeError('`max_seq_len` must be instance of `int`.')

        try:
            # Encode each sequence independently.
            # If `max_seq_len == 0`, then sequences are not padded.
            batch_token_ids = [
                self.encode(sequence, max_seq_len=max_seq_len)
                for sequence in list(batch_sequences)
            ]
        except TypeError:
            raise TypeError(
                '`batch_sequences` must be instance of `Iterable[str]`.'
            )

        # If `max_seq_len == -1`, then padded sequences to the longest sequence
        # length in the current batch. This step do not need to add `[BOS]`
        # and `[EOS]`, since `self.encode` already do the work.
        if max_seq_len == -1:
            max_seq_len = max(map(
                len,
                batch_token_ids
            ))
            batch_padding_len = map(
                lambda token_ids: max_seq_len - len(token_ids),
                batch_token_ids
            )
            batch_token_ids = list(map(
                lambda tmp: tmp[0] + [
                    self.token_to_id[self.__class__.pad_token]
                    for _ in range(tmp[1])
                ],
                zip(batch_token_ids, batch_padding_len)
            ))

        return batch_token_ids

    def build_vocab(
            self,
            batch_sequences: Iterable[str],
            min_count: int = 1
    ) -> None:
        """Build vocabulary for tokenizer.

        Vocabulary is sorted by token frenquence in descending order.

        Args:
            batch_sequences:
                Vocabulary source.
            min_count:
                Minimum of token's frequence. If token's frequence is smaller
                than `min_count`, then discard that token.
         Raises:
            TypeError:
                When `batch_sequences` is not instance of `Iterable[str]` or
                `min_count` is not instance of `int`.
        """
        # Type check.
        if not isinstance(min_count, int):
            raise TypeError('`min_count` must be instance of `int`.')

        token_freq_counter = {}

        try:
            for tokens in self.batch_sequences_to_tokens(batch_sequences):
                for token in tokens:
                    if token not in token_freq_counter:
                        token_freq_counter[token] = 0
                    token_freq_counter[token] += 1
        except TypeError:
            raise TypeError(
                '`batch_sequences` must be instance of `Iterable[str]`.'
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
