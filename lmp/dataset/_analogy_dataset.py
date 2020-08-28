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
from typing import Generator
from typing import Iterable
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
    r"""Dataset class for analogy evaluation.

    Attributes:
        samples:
            Sample for analogy test.

    Raises:
        TypeError:
            `samples` must be an instance of `Iterable[Iterable[str]]`.
        ValueError:
            Every sample must have `word_a`, `word_b`, `word_c`, `word_d` and
            `category`.
    """

    def __init__(self, samples: Iterable[Iterable[str]]):
        if not isinstance(samples, Iterable):
            raise TypeError(
                '`samples` must be an instance of `Iterable[Iterable[str]]`.'
            )

        samples = list(samples)

        if not all(map(
                lambda sample: isinstance(sample, Iterable),
                samples
        )):
            raise TypeError(
                '`samples` must be an instance of `Iterable[Iterable[str]]`.'
            )
        samples = [list(sample) for sample in samples]

        for sample in samples:
            if not all(map(
                    lambda word: isinstance(word, str),
                    sample
            )):
                raise TypeError(
                    '`samples` must be an instance of '
                    '`Iterable[Iterable[str]]`.'
                )
            if len(sample) != 5:
                raise ValueError(
                    'Every sample must have `word_a`, `word_b`, `word_c`, '
                    '`word_d` and category.'
                )

        self.samples = samples

    def __iter__(self) -> Generator[List[str], None, None]:
        r"""Iterate through each sample in the dataset.

        Yields:
            Each sample in `self.samples`.
        """
        for sample in self.samples:
            yield sample

    def __len__(self) -> int:
        r"""Dataset size."""
        return len(self.samples)

    def __getitem__(self, index: int) -> List[str]:
        r"""Sample analogy pairs by index.

        Return sample will be following format:
            (word_a, word_b, word_c, word_d, category)
        Where `word_a : word_b = word_c : word_d`, and the sample is
        categorized under `category`.

        Returns:
            word_a:
            word_b:
            word_c:
                Query words for analogy.
            word_d:
                Target word for analogy.
            category:
                Category of the sample.

        Raises:
            IndexError:
                When `index >= len(self)`.
            TypeError:
                When `index` is not an instance of `int`.
        """
        if not isinstance(index, int):
            raise TypeError('`index` must be an instance of `int`.')

        return self.samples[index]
