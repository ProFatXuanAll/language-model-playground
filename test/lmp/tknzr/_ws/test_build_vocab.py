r"""Test build_vocab operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.build_vocab`.
"""
import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
        (' ', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
        ('a a b c', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            'a': 4, 'b': 5, 'c': 6,
        }),
        ('a b c', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            'a': 4, 'b': 5, 'c': 6,
        }),
    ]
)
def test_min_count_1(test_input, expected):
    r"""tk2id must save the dictionary in represent of token to id

    If the CharTknzr initialize tk2id's value with None, it will add basic
    token([bos], [eos]...). If a token's frequency is lower than
    ``self.min_count``, then that token will not be included in the vocabulary.
    """

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=10,
        min_count=1,
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
        (' ', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
        ('a a b c', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3, 'a': 4}),
        ('a b c', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
    ]
)
def test_min_count_2(test_input, expected):
    r"""tk2id must save the dictionary in represent of token to id

    If the CharTknzr initialize tk2id's value with None, it will add basic
    token([bos], [eos]...). If a token's frequency is lower than
    ``self.min_count``, then that token will not be included in the vocabulary.
    """

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=10,
        min_count=2,
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('1 2 3', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            '1': 4, '2': 5, '3': 6
        }),
        ('a b c', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            'a': 4, 'b': 5, 'c': 6
        }),
        ('哈 囉 世 界', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            '哈': 4, '囉': 5, '世': 6, '界': 7
        }),
    ]
)
def test_max_vocab_neg1(test_input, expected):
    r"""Add as many tokens as possible to tk2id when max_vocab is -1"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('1 2 3', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
        ('a b c', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
        ('哈 囉 世 界', {'[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3}),
    ]
)
def test_max_vocab_pos1(test_input, expected):
    r"""The vocabulary size must smaller 1"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=1,
        min_count=1,
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('1 2 3', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            '1': 4, '2': 5}),
        ('a b c', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
            'a': 4, 'b': 5
        }),
        ('哈 囉 世 界', {
            '[bos]': 0, '[eos]': 1, '[pad]': 2, '[unk]': 3,
                        '哈': 4, '囉': 5
        }),
    ]
)
def test_max_vocab_pos3(test_input, expected):
    r"""The vocabulary size must smaller 3"""

    tknzr = WsTknzr(
        is_uncased=True,
        max_vocab=6,
        min_count=1,
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected
