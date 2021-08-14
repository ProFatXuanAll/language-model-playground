r"""Test token's encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.enc`.
- :py:meth:`lmp.tknzr.WsTknzr.doc`.
"""
import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            '1 2 3',
            [0, 7, 8, 3, 1],
        ),
        (
            'a b c',
            [0, 4, 5, 6, 1],
        ),
        (
            '哈 囉 世 界',
            [0, 9, 3, 3, 3, 1],
        ),
        (
            '[bos] [eos] [pad] [unk]',
            [0, 0, 1, 2, 3, 1],
        ),
    ]
)
def test_enc(tk2id, test_input, expected):
    r"""Token must be encoding to ids"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert tknzr.enc(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            [7, 8, 3],
            '1 2 [unk]',
        ),
        (
            [4, 5, 6],
            'a b c',
        ),
        (
            [9, 3, 3, 3],
            '哈 [unk] [unk] [unk]',
        ),
        (
            [0, 1, 2, 3],
            '[bos] [eos] [pad] [unk]',
        ),
    ]
)
def test_dec(tk2id, test_input, expected):
    r"""Ids must be docoding to tokens"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert tknzr.dec(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            [''],
            [[0, 1]],
        ),
        (
            ['1 2 3', 'a b c'],
            [[0, 7, 8, 3, 1],
             [0, 4, 5, 6, 1]]
        ),
        (
            ['哈 囉 世 界'],
            [[0, 9, 3, 3, 3, 1]]
        ),
        (
            ['[bos] [eos] [pad] [unk]'],
            [[0, 0, 1, 2, 3, 1]]
        ),
    ]
)
def test_batch_enc(tk2id, test_input, expected):
    r"""Turn text batch to token batch"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert tknzr.batch_enc(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            [[]],
            [''],
        ),
        (
            [[7, 8, 3], [4, 5, 6]],
            ['1 2 [unk]', 'a b c'],
        ),
        (
            [[9, 3, 3, 3]],
            ['哈 [unk] [unk] [unk]'],
        ),
        (
            [[0, 1, 2, 3]],
            ['[bos] [eos] [pad] [unk]'],
        ),
    ]
)
def test_batch_dec(tk2id, test_input, expected):
    r"""Turn token batch to token text"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert tknzr.batch_dec(test_input) == expected
