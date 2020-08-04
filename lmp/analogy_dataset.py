r"""preprocessing training dataset.

Usage:
    dataset = lmp.dataset.BaseDataset(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

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
    torch.Tensor,
]


class AnalogyDataset(torch.utils.data.Dataset):
    r"""Dataset class for word analogy."""

    def __init__(self):
        file_path = os.path.join(
            lmp.path.DATA_PATH,
            'word-test.v1.txt'
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'file {file_path} does not exist.'
            )
        with open(file_path, 'r', encoding='utf8') as txt_file:
            samples = [line.strip() for line in txt_file.readlines()]

        # Parsing.
        category = ''
        self.samples = []
        for sample in samples:
            if re.match(r'^:', sample):
                category = sample[2:]
                continue

            self.samples.append(re.split(r'\s+', sample)[:4])


    def __len__(self) -> int:
        r"""Dataset size."""
        return len(self.samples)

    def __getitem__(self, index: int) -> List[str]:
        r"""Sample single analogy pairs using index."""
        return self.samples[index]

    # @staticmethod
    # def create_collate_fn(
    #         tokenizer: lmp.tokenizer.BaseTokenizer
    # ):
    #     r"""Create `collate_fn` for `torch.utils.data.DataLoader`.

    #     Use `tokenizer` to encode tokens into tokens' ids.

    #     Attributes:
    #         tokenizer:
    #             Perform encoding.

    #     Returns:
    #         A function used by `torch.utils.data.DataLoader`.
    #     """
    #     def collate_fn(batch_analogy: List[List[str]]) -> CollateFnReturn:
    #         r"""Function used by `torch.utils.data.DataLoader`.

    #         Each analogy sample in `batch_analogy` will be first encoded by
    #         `tokenizer`, and the returned batch of tokens' ids will split
    #         into 4 groups following analogy format:
    #             word_a : word_b = word_c : word_d

    #         Returns:
    #             word_a:
    #             word_b:
    #             word_c:
    #                 Query word for analogy.
    #             word_d:
    #                 Target word for analogy.
    #         """
    #         batch_token_ids = []
    #         for analogy in batch_analogy:
    #             token_ids = []
    #             for token in analogy:
    #                 token_ids.append(
    #                     tokenizer.convert_token_to_id(token)
    #                 )
    #             batch_token_ids.append(token_ids)

    #         batch_token_ids = torch.LongTensor(batch_token_ids)

    #         # Construct sample following analogy format:
    #         word_a = batch_token_ids[:, 0]
    #         word_b = batch_token_ids[:, 1]
    #         word_c = batch_token_ids[:, 2]
    #         word_d = batch_token_ids[:, 3]

    #         return word_a, word_b, word_c, word_d

    #     return collate_fn
