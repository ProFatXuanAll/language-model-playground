r"""Test build_vocab operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.build_vocab`.
"""
import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
        # Test empty vocabulary in gerneral case
        (
            (
                True,
                -1,
                1,
                None,
            ),
            (),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
            }
        ),
        # Test Chinese characters in gerneral case
        (
            (
                True,
                -1,
                1,
                None,
            ),
            ('哈囉世界'),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                '哈': 4,
                '囉': 5,
                '世': 6,
                '界': 7,
            }
        ),
        # Test frequency in genral case
        (
            (
                True,
                -1,
                1,
                None,
            ),
            (
                'cba',
                'cb',
                'c',
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'c': 4,
                'b': 5,
                'a': 6,
            }
        ),
        # Test whitespace in general case
        (
            (
                True,
                -1,
                1,
                None,
            ),
            ('a b c'),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'a': 4,
                'b': 5,
                'c': 6,
            }
        ),
        # Test tk2id in general case
        (
            (
                True,
                -1,
                1,
                {
                    'a': 4,
                    'b': 5,
                    'c': 6,
                },
            ),
            ('abc'),
            {
                'a': 4,
                'b': 5,
                'c': 6,
            }
        ),
        # Test cased in general case
        (
            (
                True,
                -1,
                1,
                None,
            ),
            (
                'abc',
                'AB',
                'A'
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'a': 4,
                'b': 5,
                'c': 6,
            }
        ),
        # Test uncased
        (
            (
                False,
                -1,
                1,
                None,
            ),
            (
                'abc',
                'AB',
                'A'
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'A': 4,
                'a': 5,
                'b': 6,
                'c': 7,
                'B': 8,
            }
        ),
        # Test min count
        (
            (
                True,
                -1,
                2,
                None,
            ),
            (
                'abc',
                'ab',
                'a',
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'a': 4,
                'b': 5,
            }
        ),
        # Test max vocab
        (
            (
                True,
                5,
                -1,
                None,
            ),
            (
                'abc',
                'ab',
                'a',
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'a': 4,
            }
        ),
    ]
)
def test_build_vocab(parameters, test_input, expected):
    r"""tk2id must save the dictionary in represent of token to id

    If the CharTknzr initialize tk2id's value with None, it will add basic
    token([bos], [eos]...). If a token's frequency is lower than
    ``min_count``, then that token will not be included in the vocabulary.
    """

    tknzr = CharTknzr(
        is_uncased=parameters[0],
        max_vocab=parameters[1],
        min_count=parameters[2],
        tk2id=parameters[3],
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected
