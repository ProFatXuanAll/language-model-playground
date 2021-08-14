r"""Test the construction of vocabulary

Test target:
- :py:meth:`lmp.tknzr._ch.CharTknzr.build_vocab`.
"""
import pytest

from lmp.tknzr._char import CharTknzr


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
        # Test empty input sequence of text
        #
        # Expect only special tokens, when input empty text and assign
        # None for tk2id.
        (

            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
            }
        ),
        # Test Chinese characters input
        #
        # Expect the chinese characters and special tokens, when input Chinese
        # characters and tk2id with None.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
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
        # Test frequency
        #
        # Expect the higher frequency the smaller id, if they have same
        # frequency, then compare the sequence of token.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (
                'dcba',
                'dcb',
                'dc',
                'eee'
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'd': 4,
                'c': 5,
                'e': 6,
                'b': 7,
                'a': 8,
            }
        ),
        # Test whitespace
        #
        # Expect the whitespace must not be added to vocabulary.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
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
        # Test ``min_count``
        #
        # Expect only add the token whose frequency is larger than
        # ``min_count``.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 2,
                'tk2id': None,
            },
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
            },
        ),
        # Test ``max_vocab``
        #
        # Expect add the tokens until vocabulary's size is equal
        # to ``max_vocat``. If ``max_vocab`` is -1, then add all
        # token to vocabulary.
        (
            {
                'is_uncased': True,
                'max_vocab': 5,
                'min_count': 1,
                'tk2id': None,
            },
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
            },
        ),
    ]
)
def test_build_vocab(parameters, test_input, expected):
    r"""Test tk2id

    Expect tk2id must save the correct vocabulary and ids.
    """

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected
