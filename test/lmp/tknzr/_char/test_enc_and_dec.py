r"""Test enc and dec operation for token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.enc`.
- :py:meth:`lmp.tknzr.CharTknzr.doc`.
"""
import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('123', [0, 7, 8, 3, 1]),
        ('abc', [0, 4, 5, 6, 1]),
        ('哈囉世界', [0, 9, 3, 3, 3, 1]),
        ('[bos]', [0, 3, 5, 3, 3, 3, 1]),
        ('[eos]', [0, 3, 3, 3, 3, 3, 1]),
        ('[pad]', [0, 3, 3, 4, 3, 3, 1]),
        ('[unk]', [0, 3, 3, 3, 3, 3, 1]),
    ]
)
def test_enc(test_input, expected):
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

    tknz = CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=1,
                tk2id=tk2id,
            )

    assert tknz.enc(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([7, 8, 3], '12[unk]'),
        ([4, 5, 6], 'abc'),
        ([9, 3, 3, 3], '哈[unk][unk][unk]'),
        ([0, 1, 2, 3], '[bos][eos][pad][unk]'),
    ]
)
def test_dec(test_input, expected):
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

    tknz = CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=1,
                tk2id=tk2id,
            )

    assert tknz.dec(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([''], [[0, 1]]),
        (['123', 'abc'], [[0, 7, 8, 3, 1], [0, 4, 5, 6, 1]]),
        (['哈囉世界'], [[0, 9, 3, 3, 3, 1]]),
        (['[bos]'], [[0, 3, 5, 3, 3, 3, 1]]),
        (['[eos]'], [[0, 3, 3, 3, 3, 3, 1]]),
        (['[pad]'], [[0, 3, 3, 4, 3, 3, 1]]),
        (['[unk]'], [[0, 3, 3, 3, 3, 3, 1]]),
    ]
)
def test_batch_enc(test_input, expected):
    r"""Turn text batch to token batch"""

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

    tknz = CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=1,
                tk2id=tk2id,
            )

    assert tknz.batch_enc(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([[]], ['']),
        ([[7, 8, 3], [4, 5, 6]], ['12[unk]', 'abc']),
        ([[9, 3, 3, 3]], ['哈[unk][unk][unk]']),
        ([[0, 1, 2, 3]], ['[bos][eos][pad][unk]']),
    ]
)
def test_batch_dec(test_input, expected):
    r"""Turn token batch to text batch"""

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

    tknz = CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=1,
                tk2id=tk2id,
            )

    assert tknz.batch_dec(test_input) == expected
