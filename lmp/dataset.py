r"""preprocessing training dataset.

Usage:
    import lmp.dataset

    dataset = lmp.dataset.BaseDataset(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Tuple

# 3rd-party modules

import torch.utils.data

# self-made modules

import lmp.tokenizer


# Define types for type annotation.
CollateFnReturn = Tuple[torch.Tensor, torch.Tensor]
CollateFn = Callable[[Iterable[str]], CollateFnReturn]


class BaseDataset(torch.utils.data.Dataset):
    r"""Dataset class for generating language model samples.

    Attributes:
        batch_sequences:
            All sequences in the dataset.

    Raises:
        TypeError:
            When `batch_sequences` is not an instance of `Iterable[str]`.
    """

    def __init__(self, batch_sequences: Iterable[str]):
        super().__init__()
        # Type check.
        if not isinstance(batch_sequences, Iterable):
            raise TypeError(
                '`batch_sequences` must be an instance of `Iterable[str]`.'
            )

        batch_sequences = list(batch_sequences)

        if not all(map(
                lambda sequence: isinstance(sequence, str),
                batch_sequences
        )):
            raise TypeError(
                '`batch_sequences` must be an instance of `Iterable[str]`.'
            )

        self.batch_sequences = batch_sequences

    def __iter__(self) -> Generator[str, None, None]:
        r"""Iterate through each sample in the dataset.

        Yields:
            Each sequence in `self.batch_sequences`.
        """
        for sequence in self.batch_sequences:
            yield sequence

    def __len__(self) -> int:
        r"""Dataset size."""
        return len(self.batch_sequences)

    def __getitem__(self, index: int) -> str:
        r"""Sample single sequence using index.

        Raises:
            IndexError:
                When `index >= len(self)`.
            TypeError:
                When `index` is not an instance of `int`.
        """
        # Type check.
        if not isinstance(index, int):
            raise TypeError('`index` must be an instance of `int`.')

        return self.batch_sequences[index]

    @staticmethod
    def create_collate_fn(
            tokenizer: lmp.tokenizer.BaseTokenizer,
            max_seq_len: int = -1
    ) -> CollateFn:
        r"""Create `collate_fn` for `torch.utils.data.DataLoader`.

        Use `tokenizer` to perform tokenization on each mini-batch. Each
        mini-batch will be encoded into tokens' ids with length equal to
        `max_seq_len`. If `max_seq_len == -1`, then `max_seq_len` will be
        inferred from current mini-batch.

        Attributes:
            tokenizer:
                Perform both tokenization and encoding.
            max_seq_len:
                Mini-batch's maximum encoded sequence length.

        Raises:
            TypeError:
                When `tokenizer` is not an instance of
                `lmp.tokenizer.BaseTokenizer` or `max_seq_len` is not an instance
                of `int`.
            ValueError:
                When `0 <= max_seq_len <= 1` or `max_seq_len < -1`.

        Returns:
            A function used by `torch.utils.data.DataLoader`.
        """
        # Type check
        if not isinstance(tokenizer, lmp.tokenizer.BaseTokenizer):
            raise TypeError(
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.'
            )

        if not isinstance(max_seq_len, int):
            raise TypeError(
                '`max_seq_len` must be an instance of `int`.'
            )

        # Value check.
        if (0 <= max_seq_len <= 1) or (max_seq_len < -1):
            raise ValueError(
                '`max_seq_len` must be greater than `1` or equal to `-1`.'
            )

        def collate_fn(batch_sequences: Iterable[str]) -> CollateFnReturn:
            r"""Function used by `torch.utils.data.DataLoader`.

            Each sequence in `batch_sequences` will be first tokenized and
            encoded by `tokenizer`, the returned batch of tokens' ids will
            have exact same length. We construct training samples following
            language model format.

            Raises:
                TypeError:
                    When `batch_sequences` is not an instance of `Iterable[str]`.
                ValueError:
                    When `batch_sequences` is empty.

            Returns:
                x:
                    Model input batch of token's ids with numeric type
                    `torch.int64`.
                y:
                    Model predict target for each token id in `x` with numeric
                    type `torch.int64`.
            """
            if not batch_sequences:
                raise ValueError('`batch_sequences` must not be empty.')

            try:
                batch_token_ids = torch.LongTensor(
                    tokenizer.batch_encode(
                        batch_sequences,
                        max_seq_len=max_seq_len
                    )
                )

                # Construct sample following language model:
                # `batch_sequences[0][0]` must predict `batch_sequences[0][1]`,
                # `batch_sequences[0][1]` must predict `batch_sequences[0][2]`,
                # ...
                # `batch_sequences[n][m]` must predict `batch_sequences[n][m+1]`.
                x = batch_token_ids[:, :-1]
                y = batch_token_ids[:, 1:]

                return x, y
            except TypeError:
                raise TypeError(
                    '`batch_sequences` must be an instance of `Iterable[str]`.'
                )

        return collate_fn
