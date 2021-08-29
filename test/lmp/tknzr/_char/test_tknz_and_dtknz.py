r"""Test tokenization and detokenization.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.dtknz`.
- :py:meth:`lmp.tknzr.CharTknzr.tknz`.
"""

from typing import List

import pytest

import lmp.dset.util
from lmp.tknzr import CharTknzr


@pytest.mark.parametrize(
    'parameters,test_input,expected',
    [
        # Test subject:
        # Input empty string.
        #
        # Expectation:
        # Return empty list.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            '',
            [],
        ),
        # Test subject:
        # Perform case normalization on output tokens.
        #
        # Expectation:
        # Split text into lowercase characters.
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
        # Test subject:
        # Reserve case.
        #
        # Expectation:
        # Split text into characters while reserving case.
        (
            {
                'is_uncased': False,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            'ABc',
            ['A', 'B', 'c'],
        ),
        # Test subject:
        # Stripping whitespaces.
        #
        # Expectation:
        # Whitespaces are stripped from both ends.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            '  abc ',
            ['a', 'b', 'c'],
        ),
        # Test subject:
        # Reserve whitespaces between characters.
        #
        # Expectation:
        # Whitespaces are treated as single character.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            'a b  c',
            ['a', ' ', 'b', ' ', 'c'],
        ),
        # Test subject:
        # NFKC normalization.
        #
        # Expectation:
        # Perform NFKC normalization on output tokens.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            '０é',
            [lmp.dset.util.norm('０'), lmp.dset.util.norm('é')],
        ),
    ]
)
def test_tknz(parameters, test_input: str, expected: List[str]):
    r"""Tokenize text into characters."""

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
    )
    assert tknzr.tknz(test_input) == expected


@pytest.mark.parametrize(
    'parameters,test_input,expected',
    [
        # Test subject:
        # Input empty list.
        #
        # Expectation:
        # Return empty string.
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
        # Test subject:
        # Perform case normalization on output text.
        #
        # Expectation:
        # Return text in lowercase.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ['A', 'B', 'c'],
            'abc',
        ),
        # Test subject:
        # Reserve case.
        #
        # Expectation:
        # Return text in original case.
        (
            {
                'is_uncased': False,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ['A', 'B', 'c'],
            'ABc',
        ),
        # Test subject:
        # Stripping whitespaces.
        #
        # Expectation:
        # Whitespaces characters are stripped from both ends.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            [' ', 'a', 'b', 'c', ' ', ' '],
            'abc',
        ),
        # Test subject:
        # Reserve whitespaces between characters.
        #
        # Expectation:
        # Whitespaces are treated as single character.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ['a', ' ', 'b', ' ', ' ', 'c'],
            'a b c',
        ),
        # Test subject:
        # NFKC normalization.
        #
        # Expectation:
        # Perform NFKC normalization on output tokens.
        (
            {
                'is_uncased': True,
                'max_vocab': -1,
                'min_count': 1,
                'tk2id': None,
            },
            ['０', 'é'],
            lmp.dset.util.norm('０é'),
        ),
    ]
)
def test_dtknz(parameters, test_input: List[str], expected: str):
    r"""Detokenize characters back to text."""

    tknzr = CharTknzr(
        is_uncased=parameters['is_uncased'],
        max_vocab=parameters['max_vocab'],
        min_count=parameters['min_count'],
    )

    assert tknzr.dtknz(test_input) == expected
