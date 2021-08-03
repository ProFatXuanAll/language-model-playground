import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', []),
        ('1 2 3', ['1', '2', '3']),
        ('a b c', ['a', 'b', 'c']),
        ('哈囉世界', ['哈囉世界']),
    ]
)
def test_tknz(test_input, expected):
    r"""Perform tokenization on text.

    Text will be tokenize.
    """
    tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    assert tknzr.tknz(test_input) == expected
