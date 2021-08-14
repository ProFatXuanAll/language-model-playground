r"""Test the function of tokenization

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.tknz`.
- :py:meth:`lmp.tknzr.CharTknzr.dtknz`.
"""

import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
        # Test cased
        #
        # Expect the character must be transformed from capital to lower case,
        # when the `is_uncased` is true.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            'ABc',
            ['a', 'b', 'c'],
        ),
        # Test uncased
        #
        # Expect the character must not be transformed from capital
        # to lower case, when the `is_uncased` is false.
        (
            {
                'is_uncased': False,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            'ABc',
            ['A', 'B', 'c']
        ),
        (
            # Test whitespace
            #
            # Expect whitespace must be tokenenized, when they whitespace is
            # in the middle of string.
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            '12 34',
            ['1', '2', ' ', '3', '4']
        ),
        (
            # Test whitespace
            #
            # Expect whitespace must not be tokenized, when they whitespace is
            # in the front of string.
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ' 1234 ',
            ['1', '2', '3', '4']
        ),
        # Test Chinese characters input
        #
        # Expect the chinese characters be tokenized
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            '哈囉 世界',
            ['哈', '囉', ' ', '世', '界'],
        ),
    ]
)
def test_tknz(parameters, test_input, expected):
    r"""Text must be tokenize to characters"""

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
    )
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
        # Test empty input text
        #
        # Expect empty output
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            [],
            '',
        ),
        # Test whitespace
        #
        # Expect whitespace must be detokenenized, when they whitespace is
        # in the middle of string.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ['1', '2', ' ', '3'],
            '12 3',
        ),
        # Test whitespace
        #
        # Expect whitespace must not be detokenized, when they whitespace is
        # in the front of string.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            [' ', '1', '2', '3', ' '],
            '123',
        ),
        # Test Chinese characters input
        #
        # Expect the chinese characters be detokenized
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ['哈', '囉', ' ', '世', '界'],
            '哈囉 世界',
        )
    ]
)
def test_dtknz(parameters, test_input, expected):
    r"""Token must be joined by characters."""

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
    )

    assert tknzr.dtknz(test_input) == expected
