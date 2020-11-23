r"""Tokenizer base class."""

import abc
import json
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence

import lmp.path
import lmp.tknzr.util


class BaseTknzr(abc.ABC):
    r"""Tokenizer abstract base class.

    Provide basic functionality for text preprocessing, save and load
    preprocessing configurations.
    All tokenizers must inherit this base class.

    Parameters
    ==========
    is_uncased : bool
        When performing :py:meth:`lmp.tknzr.BaseTknzr.norm`, convert input
        sequence into lowercase if ``is_uncased == True``.
    max_vocab : int
        Maximum vocabulary size.
    min_count : int
        Minimum token frequency for each token to be included in tokenizer's
        vocabulary.
    tk2id : Dict[str, int], optional
        Token (a string) to id (an integer) lookup table.
        If ``tk2id is not None``, then initialize lookup table with ``tk2id``.
        Otherwise initialize lookup table with special tokens only.

    Attributes
    ==========
    bos_tk : str
        Token which represents the begining of a sequence.
        Sequences will be prepended with ``bos_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    bos_tkid : int
        Token id of ``bos_tk``.
    eos_tk : str
        Token which represents the end of a sequence.
        Sequences will be appended with ``eos_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    eos_tkid : int
        Token id of ``eos_tk``.
    file_name : str
        Tokenizer's configuration output file name.
    id2tk : Dict[int, str]
        Id (an integer) to token (a string) lookup table.
    is_uncased : bool
        When performing :py:meth:`lmp.tknzr.BaseTknzr.norm`, convert input
        sequence into lowercase if ``is_uncased == True``.
    pad_tk : str
        Token which represents paddings of a sequence.
        Sequences may be appended with ``pad_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    pad_tkid : int
        Token id of ``pad_tk``.
    tk2id : Dict[str, int]
        Token (a string) to id (an integer) lookup table.
    tknzr_name : str
        Display name for tokenizer on CLI.
        Used only for command line argument parsing.
    unk_tk : str
        Token which represents unknown tokens in a sequence.
        Tokens in sequence may be replaced with ``unk_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    unk_tkid : int
        Token id of ``unk_tk``.
    vocab_size : int
        Number of tokens in tokenizer's vocabulary.
    """
    bos_tk: str = '[bos]'
    bos_tkid: int = 0
    eos_tk: str = '[eos]'
    eos_tkid: int = 1
    file_name: str = 'tknzr.json'
    pad_tk: str = '[pad]'
    pad_tkid: int = 2
    tknzr_name: str = 'base'
    unk_tk: str = '[unk]'
    unk_tkid: int = 3

    def __init__(
            self,
            is_uncased: bool,
            max_vocab: int,
            min_count: int,
            *,
            tk2id: Optional[Dict[str, int]] = None,
    ):
        if not isinstance(is_uncased, bool):
            raise TypeError(f'is_uncased must be bool type.')
        self.is_uncased = is_uncased
        self.max_vocab = max_vocab
        self.min_count = min_count

        # Load pre-trained vocabulary.
        if tk2id is not None:
            self.tk2id = tk2id
            self.id2tk = {v: k for k, v in tk2id.items()}
        # Initialize vocabulary with special tokens.
        else:
            self.id2tk = {}
            self.tk2id = {}

            for tk, tkid in [
                [self.__class__.bos_tk, self.__class__.bos_tkid],
                [self.__class__.eos_tk, self.__class__.eos_tkid],
                [self.__class__.pad_tk, self.__class__.pad_tkid],
                [self.__class__.unk_tk, self.__class__.unk_tkid],
            ]:
                self.tk2id[tk] = tkid
                self.id2tk[tkid] = tk

    def save(self, exp_name: str) -> None:
        r"""Save tokenizer configuration in JSON format.

        This method will save the trained tokenizer's configuration into JSON
        format and named it with :py:attr:`lmp.tknzr.BaseTknzr.file_name`.

        Parameters
        ==========
        exp_name : str
            Training experiment name of the tokenizer.

        Raises
        ======
        FileExistsError
            When experiment path already exists but is not a directory.
        """
        file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
        file_path = os.path.join(file_dir, self.__class__.file_name)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        elif not os.path.isdir(file_dir):
            raise FileExistsError(f'{file_dir} is not a directory.')

        with open(file_path, 'w', encoding='utf8') as output_file:
            json.dump(
                {
                    'is_uncased': self.is_uncased,
                    'max_vocab': self.max_vocab,
                    'min_count': self.min_count,
                    'tk2id': self.tk2id,
                },
                output_file,
                ensure_ascii=False
            )

    @classmethod
    def load(cls, exp_name: str):
        r"""Load tokenizer JSON file.

        Parameters
        ==========
        exp_name
            Name of the existing experiment.

        Raises
        ======
        FileNotFoundError
            If directory `experiment` or file `experiment/tokenizer.json`
            does not exist.
        JSONDecodeError
            If tokenizer is not in JSON format.
        TypeError
            When `experiment` is not an instance of `str`.
        ValueError
            When `experiment` is empty string.
        """
        file_path = os.path.join(lmp.path.EXP_PATH, exp_name, cls.file_name)

        if not os.path.exists(file_path):
            # TODO: add run training tokenizer script hint
            raise FileNotFoundError(f'File {file_path} does not exist.')

        if os.path.isdir(file_path):
            # TODO: add remove dir and run training tokenizer script hint
            raise FileExistsError(f'{file_path} is a directory.')

        with open(file_path, 'r', encoding='utf-8') as input_file:
            return cls(**json.load(input_file))

    def norm(self, seq: str) -> str:
        r"""Normalize input sequence.

        Input sequence will first be normalized using unicode's NFKC format.
        If ``self.is_uncased == True``, then we will convert input sequence into
        lower cases. Finally we stripped both leading and trailing whitespace
        characters and convert all consecutive whitespace characters into
        single whitespace character.

        Parameters
        ==========
        seq
            Input sequence to be normalized.

        Raises
        ======
        TypeError
            When `sequence` is not an instance of `str`.

        Returns
        =======
        Normalized input sequence.
        """
        norm_seq = lmp.tknzr.util.norm(seq)
        if self.is_uncased:
            return norm_seq.lower()
        return norm_seq

    @abc.abstractmethod
    def tknz(self, seq: str) -> List[str]:
        r"""Perform tokenization on input sequence.

        Parameters
        ==========
        seq
            Input sequence to be tokenized.

        Raises
        ======
        TypeError
            When `seq` is not an instance of `str`.

        Returns
        =======
        List[str]
            Tokens represent input sequence.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `tokenize` not implemented yet.'
        )

    @abc.abstractmethod
    def dtknz(self, tks: Sequence[str]) -> str:
        r"""Convert tokens back to sequence.

        Parameters
        ==========
            tokens:
                Tokens to be converted.

        Raises
        ======
            TypeError:
                When `tokens` is not an instance of `Sequence[str]`.

        Returns
        =======
        str
            Sequence converted from input tokens.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `detokenize` not implemented yet.'
        )

    def trunc_to_max(
            self,
            tkids: Sequence[int],
            max_seq_len: int) -> List[int]:
        # Truncate to max sequence length.
        if max_seq_len == -1:
            return tkids
        return tkids[:max_seq_len]

    def pad_to_max(self, tkids: Sequence[int], max_seq_len: int) -> List[int]:
        # Calculate padding length.
        pad_len = max(0, max_seq_len - len(tkids))

        # Pad to max sequence length.
        return tkids + [self.__class__.pad_tkid] * pad_len

    def enc(self, seq: str, max_seq_len: int = -1) -> List[int]:
        r"""Encode sequence into token ids.

        Token ids have following format (presented with tokens):
            [bos] t1 t2 ... tn [eos] [pad] ... [pad]

        Returned token ids will include at least both `[bos]` and `[eos]`. In
        extremem returned token ids are exactly `[bos] [eos]` when
        `max_seq_len == 2`. This means `0 <= max_seq_len <= 1` are not allowed.

        Parameters
        ==========
        seq : str
            Sequence to be encoded.
        max_seq_len : int
            Whether to truncate or pad sequence to specified length. If
            `max_seq_len == -1`, then sequence will not be truncated or
            padded. If `max_seq_len >= 2`, then sequence will be truncated
            to length `max_seq_len` when sequence length is longer than
            `max_seq_len`; the sequence will be padded to `max_seq_len`
            when sequence length is shorter than `max_seq_len`.

        Raises
        ======
        TypeError
            When `sequence` is not an instance of `str` or `max_seq_len` is
            not an instance of `int`.
        ValueError
            When `0 <= max_seq_len <= 1` or `max_seq_len < -1`.

        Returns
        =======
        List[int]
            Token ids encoded from `sequence`.
        """
        # Prepend `[bos]` token id.
        tkids = [self.__class__.bos_tkid]
        for tk in self.tknz(seq):
            try:
                tkids.append(self.tk2id[tk])
            except KeyError:
                tkids.append(self.unk_tkid)

        # Append `[eos]` token id.
        tkids.append(self.__class__.eos_tkid)

        tkids = self.trunc_to_max(tkids=tkids, max_seq_len=max_seq_len)
        return self.pad_to_max(tkids=tkids, max_seq_len=max_seq_len)

    def dec(self, tkids: Sequence[int], rm_sp_tks: bool = False) -> str:
        r"""Decode token ids into sequence.

        Parameters
        ==========
        tkids : Sequence[int]
            Token ids to be decoded.
        rm_sp_tks : bool
            Whether to remove special tokens.
            If `rm_sp_tks == True`, then remove all special
            tokens except unknown word's token.
            See class docstring for more details on special tokens.

        Returns
        =======
        str
            Sequence decoded from `token_ids`.
        """
        if rm_sp_tks:
            sp_tkids = [
                self.__class__.bos_tkid,
                self.__class__.eos_tkid,
                self.__class__.pad_tkid,
            ]
            tkids = filter(lambda tkid: tkid not in sp_tkids, tkids)

        tks = []
        for tkid in tkids:
            try:
                tks.append(self.id2tk[tkid])
            except KeyError:
                tks.append(self.__class__.unk_tk)

        return self.dtknz(tks)

    def batch_enc(
            self,
            batch_seq: Sequence[str],
            max_seq_len: int = -1
    ) -> List[List[int]]:
        r"""Encode batch of sequence into batch of token ids.

        Each token ids in returned batch token ids will include at least both
        `[bos]` and `[eos]`. In extremem each returned token ids are exactly
        `[bos] [eos]` when `max_seq_len == 2`. This means
        `0 <= max_seq_len <= 1` are not allowed. See `encode` for each returned
        token ids' format.

        Parameters
        ==========
        batch_sequences
            Batch of sequence to be encoded.
        max_seq_len
            Whether to truncate or pad sequence to specified length. If
            `max_seq_len == -1`, then each sequence will not be truncated
            but padded to current batch's maximum sequence length. If
            `max_seq_len >= 2`, then each sequence will be truncated to
            `max_seq_len` when individual sequence length is longer than
            `max_seq_len`; each sequence will be padded to `max_seq_len`
            when individual sequence length is shorter than `max_seq_len`.

        Raises
        ======
        TypeError
            When `batch_sequences` is not an instance of `Sequence[str]` or
            `max_seq_len` is not an instance of `int`.
        ValueError
            When `0 <= max_seq_len <= 1` or `max_seq_len < -1`.

        Returns
        =======
        List[List[int]]
            Batch of token ids encoded from `batch_sequence`.
        """
        batch_tkids = [self.enc(seq=seq, max_seq_len=-1) for seq in batch_seq]

        # If `max_seq_len == -1`, then `max_seq_len` is the longest sequence
        # length in the current mini-batch. `+2` for `[bos]` and `[eos]`.
        if max_seq_len == -1:
            max_seq_len = max(map(len, batch_tkids)) + 2

        # Truncate to max sequence length.
        batch_tkids = [
            self.trunc_to_max(tkids=tkids, max_seq_len=max_seq_len)
            for tkids in batch_tkids
        ]

        # Pad to max sequence length.
        return [
            self.pad_to_max(tkids=tkids, max_seq_len=max_seq_len)
            for tkids in batch_tkids
        ]

    def batch_dec(
            self,
            batch_tkids: Sequence[Sequence[int]],
            rm_sp_tks: bool = False
    ) -> List[str]:
        r"""Decode batch of token ids into batch of sequences.

        Parameters
        ==========
        batch_tkids
            Batch of token ids to be decoded.
        rm_sp_tks
            Whether to remove special tokens. If
            `remove_special_tokens == True`, then remove all special tokens
            except unknown word's token. See class docstring for special
            tokens details.

        Raises
        ======
        TypeError
            When `batch_token_ids` is not an instance of `Sequence[Sequence[int]]` or
            `remove_special_tokens` is not an instance of `bool`.

        Returns
        =======
        List[str]
            Batch of sequence decoded from `batch_token_ids`.
        """
        return [
            self.dec(tkids=tkids, rm_sp_tks=rm_sp_tks)
            for tkids in batch_tkids
        ]

    def build_vocab(self, batch_seq: Sequence[str]) -> None:
        """Build vocabulary for tokenizer.

        Vocabulary is sorted by token's frenquency in descending order.

        Parameters
        ==========
        batch_sequences:
            Vocabulary source.
        min_count:
            Minimum of token's frequency. If token's frequency is smaller
            than `min_count`, then discard that token.

        Raises
        ======
        TypeError:
            When `batch_sequences` is not an instance of `Sequence[str]` or
            `min_count` is not an instance of `int`.
        """
        c = Counter()
        for seq in batch_seq:
            c.update(self.tknz(self.norm(seq)))

        max_id = max(self.tk2id.values()) + 1
        for tk, tk_count in c.most_common():
            # Stop adding tokens when pass vocabulary size limit.
            if max_id >= self.max_vocab:
                break

            # Skip adding the token when the token frequency is low.
            if tk_count < self.min_count:
                continue

            # Skip the token if already exists.
            if tk in self.tk2id:
                continue

            # Add token to vocabulary.
            self.tk2id[tk] = max_id
            self.id2tk[max_id] = tk
            max_id += 1

    @property
    def vocab_size(self) -> int:
        r"""Vocabulary size of tokenizer."""
        return len(self.tk2id)
