r"""Test the construction of character tokenizer's vocabulary.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.build_vocab`.
"""

from typing import Dict, Sequence

import pytest

import lmp.dset.util
from lmp.tknzr import CharTknzr


@pytest.mark.parametrize(
    'parameters,test_input,expected',
    [
        # Test subject:
        # Input empty batch of text.
        #
        # Expectation:
        # Only special tokens were added to vocabulary.
        (

            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (),
            {
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
            }
        ),
        # Test subject:
        # Unlimited vocabulary size.
        #
        # Expectation:
        # Adding all tokens into vocabulary when `max_vocab == -1`.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (
                '哈囉',
                '世界',
                'abcd',
            ),
            {
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                '哈': 4,
                '囉': 5,
                '世': 6,
                '界': 7,
                'a': 8,
                'b': 9,
                'c': 10,
                'd': 11,
            }
        ),
        # Test subject:
        # Token frequencies affect the order of construction of vocabulary.
        #
        # Expectation:
        # The higher of the token frequency, the smaller of the token id.
        # If tokens have the same frequencies, then tokens are added to
        # vocabulary in the order of appearance in `batch_txt`.
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
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                'd': 4,
                'c': 5,
                'e': 6,
                'b': 7,
                'a': 8,
            }
        ),
        # Test subject:
        # Filter tokens by `min_count`.
        #
        # Expectation:
        # Only add tokens, whose frequencies are larger than or equal to
        # `min_count`, into vocabulary.
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
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                'a': 4,
                'b': 5,
            },
        ),
        # Test subject:
        # Maximum vocabulary size.
        #
        # Expectation:
        # Keep adding tokens until vocabulary's size is equal to `max_vocab`.
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
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                'a': 4,
            },
        ),
        # Test subject:
        # Build vocabulary with normalized text.
        #
        # Expectation:
        # Vocabulary must be normalized.
        (
            {
                'is_uncased': False,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (
                '０',
                '０é',
            ),
            {
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                lmp.dset.util.norm('０'): 4,
                lmp.dset.util.norm('é'): 5,
            },
        ),
        # Test subject:
        # Differentiate upper cases and lower cases.
        #
        # Expectation:
        # Treat cases differently when `is_uncased == False`
        (
            {
                'is_uncased': False,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            (
                'ABC',
                'abc',
            ),
            {
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                'A': 4,
                'B': 5,
                'C': 6,
                'a': 7,
                'b': 8,
                'c': 9,
            },
        ),
        # Test subject:
        # Extend vocabulary.
        #
        # Expectation:
        # Build vocabulary based on existed vocabulary.
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
            (
                'def',
                'de',
                'd',
            ),
            {
                CharTknzr.bos_tk: CharTknzr.bos_tkid,
                CharTknzr.eos_tk: CharTknzr.eos_tkid,
                CharTknzr.pad_tk: CharTknzr.pad_tkid,
                CharTknzr.unk_tk: CharTknzr.unk_tkid,
                'a': 4,
                'b': 5,
                'c': 6,
                'd': 7,
                'e': 8,
                'f': 9,
            },
        ),
    ]
)
def test_build_vocab(
    parameters,
    test_input: Sequence[str],
    expected: Dict[str, int],
):
    r"""Correctly build vocabulary under the constraint of given parameters."""

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
        tk2id=parameters['tk2id'],
    )

    tknzr.build_vocab(test_input)

    assert tknzr.tk2id == expected
