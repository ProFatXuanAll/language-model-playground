import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        (' ', []),
        ('abc', ['a', 'b', 'c']),
        ('Abc', ['a', 'b', 'c']),
    ]
)
def test_cased(test_input, expected):
    r"""Transform the token from Capital case to lower case"""

    tknzr = CharTknzr(is_uncased=True, max_vocab=10, min_count=1)
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        (' ', []),
        ('abc', ['a', 'b', 'c']),
        ('Abc', ['A', 'b', 'c']),
    ]
)
def test_uncased(test_input, expected):
    r"""Ignore the transformation from Capital case to lower case"""

    tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=1)
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        (' ', []),
        ('123', ['1', '2', '3']),
        ('123  ', ['1', '2', '3']),
        (' 123', ['1', '2', '3']),
        ('abc', ['a', 'b', 'c']),
        ('哈囉世界', ['哈', '囉', '世', '界']),
        ('哈囉 世界', ['哈', '囉', ' ', '世', '界']),
    ]
)
def test_tknz(test_input, expected):
    r"""Text must be tokenize to characters"""

    tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([], ''),
        ([' '], ''),
        (['1', '2', ' ', '3'], '12 3'),
        (['1', '2', '3', ' '], '123'),
        ([' ', '1', '2', '3'], '123'),
        (['a', 'b', 'c'], 'abc'),
        (['哈', '囉', '世', '界'], '哈囉世界'),
        (['哈', '囉', ' ', '世', '界'], '哈囉 世界')
    ]
)
def test_dtknz(test_input, expected):
    r"""Token must be joined by characters."""

    tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.dtknz(test_input) == expected
