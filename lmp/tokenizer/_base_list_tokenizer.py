r"""Tokenizer base class using `list` structure.

Usage:
    class CustomTokenizer(BaseListTokenizer):
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


class BaseListTokenizer(BaseTokenizer):
    r"""Tokenizer base class using `list` structure.

    Design philosophy:
        Using `list` structure is to perform token ids lookup is faster compare
        to `dict` because python use array of pointer to implement `list`. But
        we also use the same `list` to perform inverse look up, so in theory
        `decode` is much slower compare to `encode`. But this means
        `BaseListTokenizer` will consume much lower memory compare to
        `BaseDictTokenizer` implementation.

    Attributes:
        bos_token:
            Token represent the begining of a sequence.
            Sequences will be encoded into following format:
            [BOS] t1 t2 ... tn [EOS].
        eos_token:
            Token represent the end of a sequence.
            Sequences will be encoded into following format:
            [BOS] t1 t2 ... tn [EOS].
        is_uncased:
            Whether to differentiate upper cases and lower cases.
        pad_token:
            Padding token.
            Only used when sequence length is shorter than must.
        token_to_id:
            Token to id look up data structure.
            Implemented with `list` data structure.
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

        Using `list` structure to implement token look up.
        """
        # Declare vocabulary data structure with `list` and initialize special
        # tokens mapping. `token_to_id` serves both token's id look up and
        # inverse look up.
        self.token_to_id: List = list(self.__class__.special_tokens())

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

        Returns:
            Token's id look up result. If `token` does not exist in tokenizer's
            vocabulary, then return unknown word token's id.
        """
        try:
            return self.token_to_id.index(token)
        except ValueError:
            return self.token_to_id.index(self.__class__.unk_token)

    def convert_id_to_token(self, token_id: int) -> str:
        r"""Perform token id inverse look up.

        Args:
            token_id:
                Inverse look up token's id.

        Returns:
            Token id's inverse lookup result. If `token_id` does not exist in
            tokenizer's vocabulary, then return unknown word token.
        """
        try:
            return self.token_to_id[token_id]
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

        Returns:
            Token ids encoded from `sequence`.
        """
        token_ids = self.convert_tokens_to_ids(self.tokenize(sequence))

        # Truncate to max sequence length,
        # -2 for `[BOS]` and `[EOS]`.
        if max_seq_len > 0:
            token_ids = token_ids[:max_seq_len - 2]

        # Encode token_ids with `[BOS]` and `[EOS]`.
        token_ids = [
            self.token_to_id.index(self.__class__.bos_token),
            *token_ids,
            self.token_to_id.index(self.__class__.eos_token)
        ]

        # Calculate padding length.
        padding_len = max(0, max_seq_len - len(token_ids))

        # Pad to max sequence length.
        return token_ids + [
            self.token_to_id.index(self.__class__.pad_token)
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

        Returns:
            Sequence decoded from `token_ids`.
        """
        if remove_special_tokens:
            # Get special tokens' ids except unknown token.
            special_token_ids = list(map(
                self.token_to_id.index,
                filter(
                    lambda token: token != self.__class__.unk_token,
                    self.__class__.special_tokens()
                )
            ))
            # Filter out special tokens' ids
            # and keep unknown token ids if presented.
            token_ids = list(filter(
                lambda token_id: token_id not in special_token_ids,
                token_ids
            ))

        sequence = self.detokenize(self.convert_ids_to_tokens(token_ids))

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

        Returns:
            Batch of token ids encoded from `batch_sequence`.
        """
        # Encode each sequence independently.
        # If `max_seq_len == 0`, then sequences are not padded.
        batch_token_ids = [
            self.encode(sequence, max_seq_len=max_seq_len)
            for sequence in batch_sequences
        ]

        # If `max_seq_len == 0`, then padded sequences to the longest sequence
        # length in the current batch. This step do not need to add `[BOS]`
        # and `[EOS]`, since `self.encode` already do the work.
        if max_seq_len == -1:
            max_seq_len = max(map(
                len,
                batch_token_ids
            ))
            batch_padding_len = list(map(
                lambda token_ids: max_seq_len - len(token_ids),
                batch_token_ids
            ))
            batch_token_ids = list(map(
                lambda tmp: tmp[0] + [
                    self.token_to_id.index(self.__class__.pad_token)
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
        """
        # Convert upper cases into lower cases.
        if self.is_uncased:
            batch_sequences = [
                sequence.lower()
                for sequence in batch_sequences
            ]

        token_freq_counter = {}

        for tokens in self.batch_sequences_to_tokens(batch_sequences):
            for token in tokens:
                if token not in token_freq_counter:
                    token_freq_counter[token] = 0
                token_freq_counter[token] += 1

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

        # Add new tokens to vocabulary.
        for new_token in build_vocab_iterator:
            self.token_to_id.append(new_token)
