r"""Character tokenizer using `dict` structure.

Usage::

    from lmp.tknzr import CharTknzr

    batch_sequences = (
        'I like apple.',
        'I really like to eat apple.'
    )

    tokenizer = CharTknzr()
    tokenizer.build_vocab(batch_sequences)

    sequence = batch_sequences[0]

    tokens = tokenizer.tokenize(sequence)
    sequence = tokenizer.detokenize(tokens)

    token_ids = tokenizer.encode(seqeunce)
    sequence = tokenizer.decode(token_ids)

    batch_token_ids = tokenizer.batch_encode(batch_seqeunces)
    batch_sequences = tokenizer.batch_decode(batch_token_ids)
"""


from typing import List, Sequence

from lmp.tknzr._base_tknzr import BaseTknzr


class CharTknzr(BaseTknzr):
    r"""Character tokenizer using `dict` structure.

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
    bos_tk : ClassVar[str]
        Token which represents the begining of a sequence.
        Sequences will be prepended with ``bos_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    bos_tkid : ClassVar[int]
        Token id of ``bos_tk``.
    eos_tk : ClassVar[str]
        Token which represents the end of a sequence.
        Sequences will be appended with ``eos_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    eos_tkid : ClassVar[int]
        Token id of ``eos_tk``.
    file_name : ClassVar[str]
        Tokenizer's configuration output file name.
    id2tk : Dict[int, str]
        Id (an integer) to token (a string) lookup table.
    is_uncased : bool
        When performing :py:meth:`lmp.tknzr.BaseTknzr.norm`, convert input
        sequence into lowercase if ``is_uncased == True``.
    max_vocab : int
        Maximum vocabulary size.
    min_count : int
        Minimum token frequency for each token to be included in tokenizer's
        vocabulary.
    pad_tk : ClassVar[str]
        Token which represents paddings of a sequence.
        Sequences may be appended with ``pad_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    pad_tkid : ClassVar[int]
        Token id of ``pad_tk``.
    tk2id : Dict[str, int]
        Token (a string) to id (an integer) lookup table.
    tknzr_name : ClassVar[str]
        Display name for tokenizer on CLI.
        Used only for command line argument parsing.
    unk_tk : ClassVar[str]
        Token which represents unknown tokens in a sequence.
        Tokens in sequence may be replaced with ``unk_tk`` when encoded by
        :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    unk_tkid : ClassVar[int]
        Token id of ``unk_tk``.
    vocab_size : int
        Number of tokens in tokenizer's vocabulary.

    Raises
    ======
    TypeError
        When parameters are not confront their respective type annotation.
    """

    def tokenize(self, seq: str) -> List[str]:
        r"""Perform tokenization on input sequence.

        Input sequence will first be normalized by
        `lmp.tknzr.BaseTknzr.normalize(sequence)`, then be splitted
        into tokens by `list(sequence)`. See
        `lmp.tknzr.BaseTknzr.normalize` for details on normalization
        process.

        Parameters
        ==========
        seq : str
            Input sequence to be tokenized.

        Raises
        ======
            TypeError:
                When `sequence` is not an instance of `str`.

        Returns
        =======
            Tokens (characters) represent input sequence.
        """
        try:
            # First do normalization, then perform tokenization.
            return list(self.norm(seq))
        except TypeError:
            raise TypeError('`seq` must be an instance of `str`.')

    def detokenize(self, tks: Sequence[str]) -> str:
        r"""Convert tokens back to sequence.

        Since each tokens are originally tokenized as characters, we can simply
        join them into single sequence. Output sequence will be normalized
        by `lmp.tknzr.BaseTknzr.normalize(sequence)`. See
        `lmp.tknzr.BaseTknzr.normalize` for details on normalization
        process.

        Parameters
        ==========
            tokens:
                Tokens to be converted.

        Raises
        ======
            TypeError:
                When `tokens` is not an instance of `Iterable[str]`.

        Returns
        =======
            Sequence converted from input tokens.
        """
        # Type check.
        if not isinstance(tks, Sequence):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        if not all(map(lambda token: isinstance(tks, str), tks)):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        # First perform detokenization, then do normalization.
        return self.norm(''.join(tks))
