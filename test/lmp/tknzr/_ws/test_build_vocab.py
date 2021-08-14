r"""Test the construction of vocabulary

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr`.
"""
import pytest

from lmp.tknzr._ws import WsTknzr


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
            ('哈 囉 世 界'),
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
                'cc d b a',
                'cc d b',
                'cc d',
            ),
            {
                '[bos]': 0,
                '[eos]': 1,
                '[pad]': 2,
                '[unk]': 3,
                'cc': 4,
                'd': 5,
                'b': 6,
                'a': 7,
            }
        ),
        # Test whitespace
        #
        # Expect the multiple whitespace must not influence the output.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (' a b  c '),
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
            (
                'a b c',
                'A B',
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
            (
                'a b c',
                'A B',
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
        # Test ``min_count``
        #
        # Expect only add the token whose frequency is larger
        # than ``min_count``.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 2,
                'tk2id': None,
            },
            (
                'a b c',
                'a b',
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
        # Test ``max_vocab``
        #
        # Expect add the tokens until vocabulary's size is equal
        # to ``max_vocab``. If ``max_vocab`` is -1, then add all
        # token to vocabulary.
        (
            {
                'is_uncased': True,
                'max_vocab': 5,
                'min_count': 1,
                'tk2id': None,
            },
            (
                'a b c',
                'a b',
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
    r"""Test tk2id

    Expect tk2id must save the correct vocabulary and ids.
    """

    tknzr = WsTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected
