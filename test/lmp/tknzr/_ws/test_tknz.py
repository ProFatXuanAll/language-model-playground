import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        ('1 2   3', ['1', '2', '3']),
        ('a b c', ['a', 'b', 'c']),
        ('  a b c', ['a', 'b', 'c']),
        ('a b c   ', ['a', 'b', 'c']),
        ('哈囉 世界', ['哈囉', '世界']),
        ('哈囉  世界 ', ['哈囉', '世界']),
    ]
)
def test_tknz(test_input, expected):
    r"""Perform tokenization on text."""

    tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([], ''),
        (['1', '2', ' ', '3'], '1 2 3'),
        (['a', 'b', 'c'], 'a b c'),
        (['a', 'b', 'c', ' '], 'a b c'),
        ([' ', ' ', 'a', 'b', 'c'], 'a b c'),
        (['哈囉', '世界'], '哈囉 世界'),
        (['哈囉', '世界 '], '哈囉 世界'),
    ]
)
def test_dtknz(test_input, expected):
    r"""Convert token to text."""

    tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.dtknz(test_input) == expected
