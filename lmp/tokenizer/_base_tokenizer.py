r"""Tokenizer base class.

Usage:
    class CustomTokenizer(BaseTokenizer):
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
from typing import Union

# self-made modules

import lmp.path


class BaseTokenizer:
    r"""Tokenizer base class.

    All tokenizers must inherit this base class.

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

    bos_token: str = '[BOS]'
    eos_token: str = '[EOS]'
    pad_token: str = '[PAD]'
    unk_token: str = '[UNK]'

    def __init__(self, is_uncased: bool = False):
        self.is_uncased = is_uncased
        self.reset_vocab()

        # Any class inherit `BaseTokenizer` must define `self.token_to_id` in
        # `reset_vocab`. Set this to `NONE` indicate `self.token_to_id` still
        # not implemented.
        if 'token_to_id' not in self.__dict__:
            self.token_to_id: Union[List, Dict, None] = None

    @staticmethod
    def special_tokens():
        r"""Iterating special tokens.

        This static method must be updated when adding more special tokens.

        Yields:
            All special tokens defined in class attributed.
        """
        yield BaseTokenizer.bos_token
        yield BaseTokenizer.eos_token
        yield BaseTokenizer.pad_token
        yield BaseTokenizer.unk_token

    @abc.abstractmethod
    def reset_vocab(self):
        r"""Reset vocabulary to initial state.

        This method must declare `self.token_to_id`.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `reset_vocab` not implemented yet.'
        )

    @classmethod
    @abc.abstractmethod
    def load(cls, experiment: str):
        r"""Load tokenizer JSON file.

        Args:
            experiment:
                Name of the existing experiment.

        Raises:
            ValueError:
                If `experiment` is not type `str`.
            FileNotFoundError:
                If directory `experiment` or file `experiment/tokenizer.json`
                does not exist.
            JSONDecodeError:
                If tokenizer is not in JSON format.
        """
        raise NotImplementedError(
            f'In class `{cls.__name__}`: '
            'function `load` not implemented yet.'
        )

    def save(self, experiment: str):
        r"""Save tokenizer into JSON file.

        Args:
            experiment:
                Name of the current experiment.
        """

        file_dir = os.path.join(
            lmp.path.DATA_PATH,
            experiment
        )
        file_path = os.path.join(
            file_dir,
            'tokenizer.json'
        )

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(file_path, 'w', encoding='utf8') as output_file:
            json.dump(
                self.token_to_id,
                output_file,
                ensure_ascii=False
            )

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

    def batch_sequences_to_tokens(
            self,
            batch_sequences: List[str]
    ) -> List[List[str]]:
        r"""Perform tokenization on batch of sequences.

        Args:
            batch_sequences:
                Batch of sequences to be tokenized.

        Returns:
            Batch of tokens converted from `batch_sequences`.
        """
        return [self.tokenize(sequence) for sequence in batch_sequences]

    def batch_tokens_to_sequences(
            self,
            batch_tokens: List[List[str]]
    ) -> List[str]:
        r"""Convert batch of tokens back to sequences.

        Args:
            batch_tokens:
                Batch of tokens to be converted.

        Returns:
            Batch of sequences converted from `batch_tokens`.
        """
        return [self.detokenize(tokens) for tokens in batch_tokens]

    @abc.abstractmethod
    def convert_token_to_id(self, token: str) -> int:
        r"""Perform token id look up.

        Args:
            token:
                Look up input token.

        Returns:
            Token's id look up result. If `token` does not exist in tokenizer's
            vocabulary, then return unknown word token's id.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `convert_token_to_id` not implemented yet.'
        )

    @abc.abstractmethod
    def convert_id_to_token(self, token_id: int) -> str:
        r"""Perform token id inverse look up.

        Args:
            token_id:
                Inverse look up token's id.

        Returns:
            Token id's inverse lookup result. If `token_id` does not exist in
            tokenizer's vocabulary, then return unknown word token.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `convert_id_to_token` not implemented yet.'
        )

    def convert_tokens_to_ids(
            self,
            tokens: List[str]
    ) -> List[int]:
        r"""Perform token id look up on input tokens.

        Args:
            tokens:
                Look up input tokens.

        Returns:
            Input tokens' ids.
        """
        return [
            self.convert_token_to_id(token)
            for token in tokens
        ]

    def convert_ids_to_tokens(
            self,
            token_ids: List[int]
    ) -> List[str]:
        r"""Perform token id inverse look up on input tokens' ids.

        Args:
            token_id:
                Inverse look up input tokens' ids.

        Returns:
            Tokens converted from input tokens' ids.
        """
        return [
            self.convert_id_to_token(token_id)
            for token_id in token_ids
        ]

    def batch_tokens_to_ids(
            self,
            batch_tokens: List[List[str]]
    ) -> List[List[int]]:
        r"""Perform token id look up on batch of tokens.

        Args:
            batch_tokens:
                Look up batch of tokens.

        Returns:
            Batch of tokens' ids.
        """
        return [
            self.convert_tokens_to_ids(tokens)
            for tokens in batch_tokens
        ]

    def batch_ids_to_tokens(
            self,
            batch_token_ids: List[List[int]]
    ) -> List[List[str]]:
        r"""Perform token id inverse look up on batch of tokens' ids.

        Args:
            batch_token_id:
                Inverse look up batch tokens' ids.

        Returns:
            Batch of Tokens.
        """
        return [
            self.convert_ids_to_tokens(token_ids)
            for token_ids in batch_token_ids
        ]

    def batch_sequences_to_ids(
            self,
            batch_sequences: List[str]
    ) -> List[List[int]]:
        r"""Perform token id look up on batch of sequences.

        Args:
            batch_sequences:
                Batch of sequences to be tokenized and looked up.

        Returns:
            Batch of tokens' ids.
        """
        return self.batch_tokens_to_ids(
            self.batch_sequences_to_tokens(batch_sequences)
        )

    def batch_ids_to_sequences(
            self,
            batch_token_ids: List[List[int]]
    ) -> List[str]:
        r"""Perform token id inverse look up on batch of sequences.

        Args:
            batch_token_ids:
                Batch of tokens' ids to be inverse looked up and detokenized.

        Returns:
            Batch of sequences.
        """
        return self.batch_tokens_to_sequences(
            self.batch_ids_to_tokens(batch_token_ids)
        )

    @abc.abstractmethod
    def encode(
            self,
            sequence: str,
            max_seq_len: int = 0
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
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `encode` not implemented yet.'
        )

    @abc.abstractmethod
    def decode(
            self,
            token_ids: List[int],
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
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `decode` not implemented yet.'
        )

    @abc.abstractmethod
    def batch_encode(
            self,
            batch_sequences: List[str],
            max_seq_len: int = 0
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
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `batch_encode` not implemented yet.'
        )

    def batch_decode(
            self,
            batch_token_ids: List[List[int]],
            remove_special_tokens: bool = False
    ) -> List[str]:
        r"""Decode batch of token ids into batch of sequence.

        Args:
            batch_token_ids:
                Batch of token ids to be decoded.
            remove_special_tokens:
                Whether to remove special tokens.
                If `remove_special_tokens == True`, then remove all special
                tokens except unknown word's token.
                See class docstring for more details on special tokens.

        Returns:
            Batch of sequence decoded from `batch_token_ids`.
        """

        return [
            self.decode(token_ids, remove_special_tokens=remove_special_tokens)
            for token_ids in batch_token_ids
        ]

    @abc.abstractmethod
    def build_vocab(
            self,
            batch_sequences: List[str],
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
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'function `build_vocab` not implemented yet.'
        )

    @property
    def vocab_size(self) -> int:
        r"""Vocabulary size of tokenizer."""
        return len(self.token_to_id)
