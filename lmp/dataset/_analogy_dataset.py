r"""Dataset for analogy evaluation.

Usage:
    import lmp.dataset

    dataset = lmp.dataset.AnalogyDataset(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

from typing import Callable
from typing import List
from typing import Tuple

# 3rd-party modules

import torch.utils.data

# self-made modules

import lmp.tokenizer
import lmp.dataset
import lmp.path

# Define types for type annotation.
CollateFnReturn = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
    str,
]
CollateFn = Callable[
    [List[List[str]]],
    CollateFnReturn
]


class AnalogyDataset(torch.utils.data.Dataset):
    r"""Dataset class for analogy evaluation."""

    def __init__(self,samples):
        self.samples=samples

    def __len__(self) -> int:
        r"""Dataset size."""
        return len(self.samples)

    def __getitem__(self, index: int) -> List[str]:
        r"""Sample single analogy pairs using index.

        Returns:
            word_a:
            word_b:
            word_c:
            word_d:
            category:
        """
        return self.samples[index]

    @staticmethod
    def create_collate_fn(
            tokenizer: lmp.tokenizer.BaseTokenizer
    ) -> CollateFn:
        r"""Create `collate_fn` for `torch.utils.data.DataLoader`.

        Use `tokenizer` to encode tokens into tokens' ids.

        Attributes:
            tokenizer:
                Perform encoding.

        Returns:
            A function used by `torch.utils.data.DataLoader`.
        """
        def collate_fn(batch_analogy: List[List[str]]) -> CollateFnReturn:
            r"""Function used by `torch.utils.data.DataLoader`.

            Each analogy sample in `batch_analogy` will be first encoded by
            `tokenizer`, and the returned batch of tokens' ids will split
            into 4 groups following analogy format:
                word_a : word_b = word_c : word_d

            Returns:
                word_a:
                word_b:
                word_c:
                    Query word for analogy.
                word_d:
                    Target word for analogy.
            """
            batch_token_ids = []
            for analogy in batch_analogy:
                token_ids = []
                for token in analogy[:-2]:
                    token_ids.append(
                        tokenizer.convert_token_to_id(token)
                    )
                batch_token_ids.append(token_ids)

            batch_token_ids = torch.LongTensor(batch_token_ids)

            # Construct sample following analogy format:
            word_a_id = batch_token_ids[:, 0]
            word_b_id = batch_token_ids[:, 1]
            word_c_id = batch_token_ids[:, 2]
            word_d = [analogy[-2] for analogy in batch_analogy]
            category = [analogy[-1] for analogy in batch_analogy]

            return word_a_id, word_b_id, word_c_id, word_d, category

        return collate_fn
