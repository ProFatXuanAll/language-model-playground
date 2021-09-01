r"""Test tokenizer's vocab size.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.vocab_size`.
"""
import pytest

from lmp.tknzr._char import CharTknzr


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
        # Automatically calculate `vocab_size`.
        #
        # Expectation:
        # Count the length of tk2id.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': {
                    CharTknzr.bos_tk: CharTknzr.bos_tkid,
                    CharTknzr.eos_tk: CharTknzr.eos_tkid,
                    CharTknzr.pad_tk: CharTknzr.pad_tkid,
                    CharTknzr.unk_tk: CharTknzr.unk_tkid,
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
    expected,
):
    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    # Check the type of `vocab_size`.
    assert isinstance(tknzr.vocab_size, int)

    # Check the value of `vocab_size`.
    assert tknzr.vocab_size == expected
