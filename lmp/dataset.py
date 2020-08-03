r"""preprocessing training dataset.

Usage:
    dataset = lmp.dataset.BaseDataset(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List
from typing import Tuple

# 3rd-party modules

import torch.utils.data

# self-made modules

import lmp.tokenizer


# Define types for type annotation.
CollateFnReturn = Tuple[
    torch.Tensor,
    torch.Tensor
]


class BaseDataset(torch.utils.data.Dataset):
    r"""Dataset class for generating language model samples.

    Attributes:
        batch_sequences:
            All sequences in the dataset.
    """

    def __init__(
            self,
            batch_sequences: List[str]
    ):
        super().__init__()
        self.batch_sequences = batch_sequences

    def __iter__(self):
        r"""Iterate each sample in the dataset."""
        for sequence in self.batch_sequences:
            yield sequence

    def __len__(self) -> int:
        r"""Dataset size."""
        return len(self.batch_sequences)

    def __getitem__(self, index: int) -> str:
        r"""Sample single sequence using index."""
        return self.batch_sequences[index]

    @staticmethod
    def create_collate_fn(
            tokenizer: lmp.tokenizer.BaseTokenizer,
            max_seq_len: int = -1
    ):
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

        Returns:
            A function used by `torch.utils.data.DataLoader`.
        """
        def collate_fn(batch_sequences: List[str]) -> CollateFnReturn:
            r"""Function used by `torch.utils.data.DataLoader`.

            Each sequence in `batch_sequences` will be first tokenized and
            encoded by `tokenizer`, the returned batch of tokens' ids will
            have exact same length. We construct training samples following
            language model format.

            Returns:
                x:
                    Model input batch of token's ids with numeric type
                    `torch.int64`.
                y:
                    Model predict target for each token id in `x` with numeric
                    type `torch.int64`.
            """
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

        return collate_fn
