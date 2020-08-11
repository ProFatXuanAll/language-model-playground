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

from typing import Generator
from typing import Iterable
from typing import List

# self-made modules

import lmp.path


class BaseTokenizer:
    r"""Tokenizer base class.

    All tokenizers must inherit this base class.

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
            Token to id look up data structure.
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
    bos_token: str = '[BOS]'
    eos_token: str = '[EOS]'
    pad_token: str = '[PAD]'
    unk_token: str = '[UNK]'

    def __init__(self, is_uncased: bool = False):
        # Type check.
        if not isinstance(is_uncased, bool):
            raise TypeError('`is_uncased` must be an instance of `bool`.')

        self.is_uncased = bool(is_uncased)

        # Any class inherit `BaseTokenizer` must define instance attribute
        # `token_to_id` in method `reset_vocab`.
        self.reset_vocab()

    @staticmethod
    def special_tokens() -> Generator[str, None, None]:
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
    def reset_vocab(self) -> None:
        r"""Reset vocabulary to initial state.

        This method must declare `self.token_to_id`.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `reset_vocab` not implemented yet.'
        )

    @classmethod
    @abc.abstractmethod
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
        raise NotImplementedError(
            f'In class `{cls.__name__}`: '
            'class method `load` not implemented yet.'
        )

    def save(self, experiment: str) -> None:
        r"""Save tokenizer into JSON file.

        Args:
            experiment:
                Name of the current experiment.

        Raises:
            FileExistsError:
                When experiment path already exists but is not a directory.
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

        file_dir = os.path.join(lmp.path.DATA_PATH, experiment)
        file_path = os.path.join(file_dir, 'tokenizer.json')

        create_dir_flag = False
        create_file_flag = False

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            create_dir_flag = True

        elif not os.path.isdir(file_dir):
            raise FileExistsError(f'{file_dir} is not a directory.')

        try:
            with open(file_path, 'w', encoding='utf8') as output_file:
                json.dump(
                    {
                        'is_uncased': self.is_uncased,
                        'token_to_id': self.token_to_id,
                    },
                    output_file,
                    ensure_ascii=False
                )
            create_file_flag = True
        except AttributeError:
            raise NotImplementedError(
                f'In class `{self.__class__.__name__}`: '
                'method `reset_vocab` not implemented yet.'
            )
        finally:
            if not create_file_flag:
                if os.path.exists(file_path):
                    os.remove(file_path)

                if create_dir_flag and os.path.exists(file_dir):
                    os.removedirs(file_dir)

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

    @abc.abstractmethod
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
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `convert_token_to_id` not implemented yet.'
        )

    @abc.abstractmethod
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
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `convert_id_to_token` not implemented yet.'
        )

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        r"""Perform tokens id look up.

        Args:
            tokens:
                Look up input tokens.

        Raises:
            TypeError:
                When `tokens` is not an instance of `Iterable[str]`.

        Returns:
            Input tokens' ids.
        """
        if not isinstance(tokens, Iterable):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        try:
            return [
                self.convert_token_to_id(token)
                for token in tokens
            ]
        except TypeError:
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

    def convert_ids_to_tokens(self, token_ids: Iterable[int]) -> List[str]:
        r"""Perform tokens id' inverse look up.

        Args:
            token_id:
                Inverse look up input tokens' id.

        Raises:
            TypeError:
                When `token_ids` is not an instance of `Iterable[int]`.

        Returns:
            Tokens converted from input tokens' id.
        """
        if not isinstance(token_ids, Iterable):
            raise TypeError(
                '`token_ids` must be an instance of `Iterable[int]`.'
            )

        try:
            return [
                self.convert_id_to_token(token_id)
                for token_id in token_ids
            ]
        except TypeError:
            raise TypeError(
                '`token_ids` must be an instance of `Iterable[int]`.'
            )

    def encode(self, sequence: str, max_seq_len: int = -1) -> List[int]:
        r"""Encode sequence into token ids.

        Token ids have following format (presented with tokens):
            [BOS] t1 t2 ... tn [EOS] [PAD] ... [PAD]

        Returned token ids will include at least both `[BOS]` and `[EOS]`. In
        extremem returned token ids are exactly `[BOS] [EOS]` when
        `max_seq_len == 2`. This means `0 <= max_seq_len <= 1` are not allowed.

        Args:
            sequence:
                Sequence to be encoded.
            max_seq_len:
                Whether to truncate or pad sequence to specified length. If
                `max_seq_len == -1`, then sequence will not be truncated or
                padded. If `max_seq_len >= 2`, then sequence will be truncated
                to length `max_seq_len` when sequence length is longer than
                `max_seq_len`; the sequence will be padded to `max_seq_len`
                when sequence length is shorter than `max_seq_len`.

        Raises:
            TypeError:
                When `sequence` is not an instance of `str` or `max_seq_len` is
                not an instance of `int`.
            ValueError:
                When `0 <= max_seq_len <= 1` or `max_seq_len < -1`.

        Returns:
            Token ids encoded from `sequence`.
        """
        # Type check.
        if not isinstance(max_seq_len, int):
            raise TypeError('`max_seq_len` must be an instance of `int`.')

        # Value check.
        if (0 <= max_seq_len <= 1) or (max_seq_len < -1):
            raise ValueError(
                '`max_seq_len` must be greater than `1` or equal to `-1`.'
            )

        try:
            token_ids = self.convert_tokens_to_ids(self.tokenize(sequence))
        except TypeError:
            raise TypeError('`sequence` must be an instance of `str`.')

        # Truncate to max sequence length,
        # -2 for `[BOS]` and `[EOS]`.
        if max_seq_len != -1:
            token_ids = token_ids[:max_seq_len - 2]

        # Prepend `[BOS]` and append `[EOS]`.
        token_ids = (
            [self.convert_token_to_id(self.__class__.bos_token)] +
            token_ids +
            [self.convert_token_to_id(self.__class__.eos_token)]
        )

        # Calculate padding length.
        padding_len = max(0, max_seq_len - len(token_ids))

        # Pad to max sequence length.
        return (
            token_ids +
            [self.convert_token_to_id(self.__class__.pad_token)] * padding_len
        )

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
                When `token_ids` is not an instance of `Iterable[int]` or
                `remove_special_tokens` is not an instance of `int`.

        Returns:
            Sequence decoded from `token_ids`.
        """
        # Type check.
        if not isinstance(token_ids, Iterable):
            raise TypeError(
                '`token_ids` must be an instance of `Iterable[int]`.'
            )

        if not isinstance(remove_special_tokens, bool):
            raise TypeError(
                '`remove_special_tokens` must be an instance of `bool`.'
            )

        if remove_special_tokens:
            # Get special tokens' ids except unknown token.
            special_token_ids = list(
                map(
                    self.convert_token_to_id,
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
            return self.detokenize(self.convert_ids_to_tokens(token_ids))
        except TypeError:
            raise TypeError(
                '`token_ids` must be an instance of `Iterable[int]`.'
            )

    def batch_encode(
            self,
            batch_sequences: Iterable[str],
            max_seq_len: int = -1
    ) -> List[List[int]]:
        r"""Encode batch of sequence into batch of token ids.

        Each token ids in returned batch token ids will include at least both
        `[BOS]` and `[EOS]`. In extremem each returned token ids are exactly
        `[BOS] [EOS]` when `max_seq_len == 2`. This means
        `0 <= max_seq_len <= 1` are not allowed. See `encode` for each returned
        token ids' format.

        Args:
            batch_sequences:
                Batch of sequence to be encoded.
            max_seq_len:
                Whether to truncate or pad sequence to specified length. If
                `max_seq_len == -1`, then each sequence will not be truncated
                but padded to current batch's maximum sequence length. If
                `max_seq_len >= 2`, then each sequence will be truncated to
                `max_seq_len` when individual sequence length is longer than
                `max_seq_len`; each sequence will be padded to `max_seq_len`
                when individual sequence length is shorter than `max_seq_len`.

        Raises:
            TypeError:
                When `batch_sequences` is not an instance of `Iterable[str]` or
                `max_seq_len` is not an instance of `int`.
            ValueError:
                When `0 <= max_seq_len <= 1` or `max_seq_len < -1`.

        Returns:
            Batch of token ids encoded from `batch_sequence`.
        """
        # Type check.
        if not isinstance(batch_sequences, Iterable):
            raise TypeError(
                '`batch_sequences` must be an instance of `Iterable[str]`.'
            )

        batch_sequences = list(batch_sequences)

        if not isinstance(max_seq_len, int):
            raise TypeError('`max_seq_len` must be an instance of `int`.')

        try:
            # If `max_seq_len == -1`, then `max_seq_len` is the longest sequence
            # length in the current mini-batch. `+2` for `[BOS]` and `[EOS]`.
            if max_seq_len == -1:
                max_seq_len = max([0] + list(map(
                    len,
                    [self.tokenize(sequence) for sequence in batch_sequences]
                ))) + 2

            # Encode each sequence..
            return [
                self.encode(sequence, max_seq_len=max_seq_len)
                for sequence in batch_sequences
            ]
        except TypeError:
            raise TypeError(
                '`batch_sequences` must be an instance of `Iterable[str]`.'
            )
        except ValueError:
            raise ValueError(
                '`max_seq_len` must be greater than `1` or equal to `-1`.'
            )

    def batch_decode(
            self,
            batch_token_ids: Iterable[Iterable[int]],
            remove_special_tokens: bool = False
    ) -> List[str]:
        r"""Decode batch of token ids into batch of sequences.

        Args:
            batch_token_ids:
                Batch of token ids to be decoded.
            remove_special_tokens:
                Whether to remove special tokens. If
                `remove_special_tokens == True`, then remove all special tokens
                except unknown word's token. See class docstring for special
                tokens details.

        Raises:
            TypeError:
                When `batch_token_ids` is not an instance of `Iterable[Iterable[int]]` or
                `remove_special_tokens` is not an instance of `bool`.

        Returns:
            Batch of sequence decoded from `batch_token_ids`.
        """
        # Type check.
        if not isinstance(batch_token_ids, Iterable):
            raise TypeError(
                '`batch_token_ids` must be an instance of '
                '`Iterable[Iterable[int]]`.'
            )

        try:
            return [
                self.decode(
                    token_ids,
                    remove_special_tokens=remove_special_tokens
                )
                for token_ids in batch_token_ids
            ]
        except TypeError as err:
            if 'token_ids' in err.args[0]:
                err_msg = (
                    '`batch_token_ids` must be an instance of '
                    '`Iterable[Iterable[int]]`.'
                )
            else:
                err_msg = (
                    '`remove_special_tokens` must be an instance of `bool`.'
                )

            raise TypeError(err_msg)

    @abc.abstractmethod
    def build_vocab(
            self,
            batch_sequences: Iterable[str],
            min_count: int = 1
    ) -> None:
        """Build vocabulary for tokenizer.

        Vocabulary is sorted by token's frenquency in descending order.

        Raises:
            TypeError:
                When `batch_sequences` is not an instance of `Iterable[str]` or
                `min_count` is not an instance of `int`.

        Args:
            batch_sequences:
                Vocabulary source.
            min_count:
                Minimum of token's frequency. If token's frequency is smaller
                than `min_count`, then discard that token.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `build_vocab` not implemented yet.'
        )

    @property
    def vocab_size(self) -> int:
        r"""Vocabulary size of tokenizer."""
        return len(self.token_to_id)
