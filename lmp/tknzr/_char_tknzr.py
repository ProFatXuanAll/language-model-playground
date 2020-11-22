r"""Character tokenizer using `dict` structure.

Usage:
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


from typing import Iterable, List

from lmp.tknzr._base_tknzr import BaseTknzr


class CharTknzr(BaseTknzr):
    r"""Character tokenizer using `dict` structure.

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

    Raises
        ======
        TypeError:
            When `is_uncased` is not an instance of `bool`.
    """

    def tokenize(self, sequence: str) -> List[str]:
        r"""Perform tokenization on input sequence.

        Input sequence will first be normalized by
        `lmp.tknzr.BaseTknzr.normalize(sequence)`, then be splitted
        into tokens by `list(sequence)`. See
        `lmp.tknzr.BaseTknzr.normalize` for details on normalization
        process.

        Parameters
        ==========
            sequence:
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
            return list(self.normalize(sequence))
        except TypeError:
            raise TypeError('`sequence` must be an instance of `str`.')

    def detokenize(self, tokens: Iterable[str]) -> str:
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
        if not isinstance(tokens, Iterable):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        tokens = list(tokens)

        if not all(map(lambda token: isinstance(token, str), tokens)):
            raise TypeError('`tokens` must be an instance of `Iterable[str]`.')

        # First perform detokenization, then do normalization.
        return self.normalize(''.join(tokens))
