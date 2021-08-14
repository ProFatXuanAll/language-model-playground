r"""Test token's encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.enc`.
- :py:meth:`lmp.tknzr.CharTknzr.doc`.
"""
import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            '123',
            [0, 7, 8, 3, 1],
        ),
        (
            'abc',
            [0, 4, 5, 6, 1],
        ),
        (
            '哈囉世界',
            [0, 9, 3, 3, 3, 1],
        ),
        (
            '[bos]',
            [0, 3, 5, 3, 3, 3, 1],
        ),
        (
            '[eos]',
            [0, 3, 3, 3, 3, 3, 1],
        ),
        (
            '[pad]',
            [0, 3, 3, 4, 3, 3, 1],
        ),
        (
            '[unk]',
            [0, 3, 3, 3, 3, 3, 1],
        ),
    ]
)
def test_enc(tk2id, test_input, expected):
    r"""Token must be encoding to ids"""

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
        (
            [7, 8, 3],
            '12[unk]',
        ),
        (
            [4, 5, 6],
            'abc',
        ),
        (
            [9, 3, 3, 3],
            '哈[unk][unk][unk]',
        ),
        (
            [0, 1, 2, 3],
            '[bos][eos][pad][unk]',
        ),
    ]
)
def test_dec(tk2id, test_input, expected):
    r"""Ids must be docoding to tokens"""

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
        (
            [''],
            [[0, 1]],
        ),
        (
            ['123', 'abc'],
            [[0, 7, 8, 3, 1], [0, 4, 5, 6, 1]],
        ),
        (
            ['哈囉世界'],
            [[0, 9, 3, 3, 3, 1]],
        ),
        (
            ['[bos]'],
            [[0, 3, 5, 3, 3, 3, 1]],
        ),
        (
            ['[eos]'],
            [[0, 3, 3, 3, 3, 3, 1]],
        ),
        (
            ['[pad]'],
            [[0, 3, 3, 4, 3, 3, 1]],
        ),
        (
            ['[unk]'],
            [[0, 3, 3, 3, 3, 3, 1]],
        ),
    ]
)
def test_batch_enc(tk2id, test_input, expected):
    r"""Turn text batch to token batch"""

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
        (
            [[]],
            ['']
        ),
        (
            [[7, 8, 3], [4, 5, 6]],
            ['12[unk]', 'abc']
        ),
        (
            [[9, 3, 3, 3]],
            ['哈[unk][unk][unk]']
        ),
        (
            [[0, 1, 2, 3]],
            ['[bos][eos][pad][unk]']
        ),
    ]
)
def test_batch_dec(tk2id, test_input, expected):
    r"""Turn token batch to text batch"""

    tknz = CharTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert tknz.batch_dec(test_input) == expected
