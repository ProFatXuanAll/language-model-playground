r"""Test tokenizer's vocab size.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.vocab_size`.
"""

import pytest

from lmp.tknzr import WsTknzr


@pytest.mark.parametrize(
    'parameters,expected',
    [
        # Test subject:
        # Empty input.
        #
        # Expectation:
        # Return number of special tokens.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            4,
        ),
        # Test subject:
        # Automatically calculate  vocabulary size..
        #
        # Expectation:
        # Count the length of `tk2id`.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': {
                    WsTknzr.bos_tk: WsTknzr.bos_tkid,
                    WsTknzr.eos_tk: WsTknzr.eos_tkid,
                    WsTknzr.pad_tk: WsTknzr.pad_tkid,
                    WsTknzr.unk_tk: WsTknzr.unk_tkid,
                    'a': 4,
                    'b': 5,
                    'c': 6,
                },
            },
            7,
        ),
    ],
)
def test_vocab_size(
    parameters,
    expected: int,
):
    r"""``WsTknzr.vocab_size`` is an instance property

    Value of ``WsTknzr.vocab_size`` is the number of tokens included in the
    vocabulary, thus must be a postive integer.
    """
    tknzr = WsTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    # Check the type of `vocab_size`.
    assert isinstance(tknzr.vocab_size, int)

    # Check the value of `vocab_size`.
    assert tknzr.vocab_size == expected
