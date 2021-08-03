import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        ('123', ['1', '2', '3']),
        ('abc', ['a', 'b', 'c']),
        ('哈囉世界', ['哈', '囉', '世', '界'])
    ]
)
def test_tknz(test_input, expected):
    r"""Perform tokenization on text.

    Text will be tokenize.
    """
    tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([], ''),
        (['1', '2', ' ', '3'], '12 3'),
        (['a', 'b', 'c'], 'abc'),
        (['哈', '囉', '世', '界'], '哈囉世界')
    ]
)
def test_dtknz(test_input, expected):
    r"""Convert token to text.

    Token will be simply joined to text.
    """
    tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.dtknz(test_input) == expected
