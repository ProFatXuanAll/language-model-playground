import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        (' ', []),
        ('a b c', ['a', 'b', 'c']),
        ('A b c', ['a', 'b', 'c']),
    ]
)
def test_cased(test_input, expected):
    r"""Transform the token from Capital case to lower case"""

    tknzr = WsTknzr(is_uncased=True, max_vocab=10, min_count=1)
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        (' ', []),
        ('a b c', ['a', 'b', 'c']),
        ('A b c', ['A', 'b', 'c']),
    ]
)
def test_uncased(test_input, expected):
    r"""Ignore the transformation from Capital case to lower case"""

    tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=1)
    assert tknzr.tknz(test_input) == expected


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
    r"""Text must be tokenize with whitespace"""

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
    r"""Token must be joined by whitespace."""

    tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.dtknz(test_input) == expected
