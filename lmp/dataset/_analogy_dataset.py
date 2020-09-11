r"""Word analogy dataset.

Usage:
    import lmp.dataset

    dataset = lmp.dataset.AnalogyDataset(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import List
from typing import Tuple

# 3rd-party modules

import torch.utils.data


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
    r"""Dataset class generating word analogy samples.

    Attributes:
        samples:
            Word analogy samples. Each sample must consist of 5 `str`. Both the
            first and second `str` must be a pair of words which have some
            syntatic or semantic relationship S1. Similarly both the third and
            fourth `str` must be a pair of words which have some syntatic or
            semantic relationship S2. The two word pairs must analog to each
            other, i.e., S1 = S2. The last `str` must be a description of S1.

            For example, given the tuple of 5 `str`:
                `('good', 'best', 'bad', 'worst', 'superlative')`
            Both the first and second `str` are adjective, and the second `str`
            is the superlative of the first `str`. Similarly both the third and
            fourth `str` are adjective, and the fourth `str` is the superlative
            of the third `str`. The last `str` is literally superlative.

            For the rest code and comments in this class, we will refer to each
            5 `str` as `word_a`, `word_b`, `word_c`, `word_d` and `category`.

    Raises:
        TypeError:
            `samples` must be an instance of `Iterable[Iterable[str]]`.
        ValueError:
            When some of the samples are not consist of 5 `str`.
    """

    def __init__(self, samples: Iterable[Iterable[str]]):
        # Type check.
        type_error_msg = (
            '`samples` must be an instance of `Iterable[Iterable[str]]`.'
        )
        if not isinstance(samples, Iterable):
            raise TypeError(type_error_msg)

        samples = list(samples)

        if not all(map(lambda sample: isinstance(sample, Iterable), samples)):
            raise TypeError(type_error_msg)

        samples = [list(sample) for sample in samples]

        for sample in samples:
            if not all(map(lambda word: isinstance(word, str), sample)):
                raise TypeError(type_error_msg)

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
        r"""Sample word analogy pairs by index.

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
