r"""Test the construction of :py:mod:`lmp.tknzr.WsTknzr`

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.init`.
"""

import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
        # Test tk2id
        #
        # Expect tk2id must be assigned as input.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id':
                    {
                        'a': 4,
                        'b': 5,
                        'c': 6,
                    },
            },
            ('a b c'),
            {
                'a': 4,
                'b': 5,
                'c': 6,
            }
        ),
        # Test tk2id
        #
        # Expect tk2id must be assigned special tokens when tk2id is None.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (
                'a b c',
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
    ]
)
def test_init(parameters, test_input, expected):
    r"""Test CharTknzr initialization"""

    tknzr = WsTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected
