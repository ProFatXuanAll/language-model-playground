r"""Test enc and dec operation for token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.enc`.
- :py:meth:`lmp.tknzr.BaseTknzr.doc`.
"""
from typing import Type

import pytest

from lmp.tknzr._base import BaseTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('1 2 3', [0, 7, 8, 3, 1]),
        ('a b c', [0, 4, 5, 6, 1]),
        ('哈 囉 世 界', [0, 9, 3, 3, 3, 1]),
        ('[bos] [eos] [pad] [unk]', [0, 0, 1, 2, 3, 1]),
    ]
)
def test_enc(subclss_tknzr_clss: Type[BaseTknzr], test_input, expected):
    r"""Token must be encoding to ids"""

    tk2id = {
        '[bos]': 0,
        '[eos]': 1,
        '[pad]': 2,
        '[unk]': 3,
        'a': 4,
        'b': 5,
        'c': 6,
        '1': 7,
        '2': 8,
        '哈': 9,
    }

    subclss_tknzr = subclss_tknzr_clss(
        is_uncased=True,
        max_vocab=10,
        min_count=1,
        tk2id=tk2id,
    )

    assert subclss_tknzr.enc(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([7, 8, 3], '12[unk]'),
        ([4, 5, 6], 'abc'),
        ([9, 3, 3, 3], '哈[unk][unk][unk]'),
        ([0, 1, 2, 3], '[bos][eos][pad][unk]'),
    ]
)
def test_dec(subclss_tknzr_clss: Type[BaseTknzr], test_input, expected):
    r"""Ids must be docoding to tokens"""

    tk2id = {
        '[bos]': 0,
        '[eos]': 1,
        '[pad]': 2,
        '[unk]': 3,
        'a': 4,
        'b': 5,
        'c': 6,
        '1': 7,
        '2': 8,
        '哈': 9,
    }

    subclss_tknzr = subclss_tknzr_clss(
        is_uncased=True,
        max_vocab=10,
        min_count=1,
        tk2id=tk2id,
    )

    assert subclss_tknzr.dec(test_input) == expected
