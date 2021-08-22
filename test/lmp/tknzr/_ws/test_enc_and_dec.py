r"""Test token's encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.enc`.
- :py:meth:`lmp.tknzr.WsTknzr.doc`.
"""
import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.mark.parametrize(
    "parameters",
    [
        # Test input number
        #
        # Expect add bos in front of the output and add eos in the end of
        # output and transform tokens to ids.
        (
            {
                "max_seq_len": -1,
                "test_input": '1 2 3',
                "expected": [0, 7, 8, 3, 1],
            }
        ),
        # Test input english characters
        #
        # Expect add bos in front of the output and add eos in the end of
        # output and transform tokens to ids.
        (
            {
                "max_seq_len": -1,
                "test_input": 'a b c',
                "expected": [0, 4, 5, 6, 1],
            }
        ),
        # Test input chinsese words
        #
        # Expect add bos in front of the output and add eos in the end of
        # output, and output `3` when encounter unknown characters.
        (
            {
                "max_seq_len": -1,
                "test_input": '哈 囉 世 界',
                "expected": [0, 9, 3, 3, 3, 1],
            }
        ),
        # Test `max_seq_len` is larger than test_input length
        #
        # Expect add pad_tk at the end of input sequence until test_input
        # length is `max_seq_len`
        (
            {
                "max_seq_len": 7,
                "test_input": 'a b c',
                "expected": [0, 4, 5, 6, 1, 2, 2],
            }
        ),
        # Test `max_seq_len` is smaller than test_input length
        #
        # Expect truncate the test_input length to `max_seq_len`.
        (
            {
                "max_seq_len": 7,
                "test_input": 'a b c a b c a b c',
                "expected": [0, 4, 5, 6, 4, 5, 6],
            }
        )
    ]
)
def test_enc(tk2id, parameters):
    r"""Token must be encoding to ids"""

    tknz = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert (
        tknz.enc(parameters['test_input'],
                 max_seq_len=parameters['max_seq_len'])
        ==
        parameters['expected']
    )


@pytest.mark.parametrize(
    "parameters",
    [
        # Test output special tokens
        #
        # Expect input id `3` and output special tokens.
        (

            {
                "rm_sp_tks": False,
                "test_input": [3, 2, 1, 0],
                "expected": '[unk] [pad] [eos] [bos]',
            }
        ),
        # Test decode characters
        #
        # Expect output tokens transfered from ids.
        (
            {
                "rm_sp_tks": False,
                "test_input": [4, 5, 6],
                "expected": 'a b c',
            }
        ),
        # Test remove special tokens
        #
        # Expect remove all special tokens except unknown tokens.
        (

            {
                "rm_sp_tks": True,
                "test_input": [3, 2, 1, 0],
                "expected": '[unk]',
            }
        ),
    ]
)
def test_dec(tk2id, parameters):
    r"""Ids must be docoding to tokens"""

    tknz = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert (
        tknz.dec(parameters['test_input'], rm_sp_tks=parameters['rm_sp_tks'])
        ==
        parameters['expected']
    )


@pytest.mark.parametrize(
    "parameters",
    [
        # Test empty sequence
        #
        # Expect bos in front of list and eos at the end of list.
        (
            {
                "max_seq_len": -1,
                "test_input": [''],
                "expected": [[0, 1]],
            }
        ),
        # Test input list of sequence
        #
        # Expect output ids transfered from tokens, and output same
        # length sequence in the list.
        (
            {
                "max_seq_len": -1,
                "test_input": ['1 2 3', 'a b c', '哈 囉 世 界'],
                "expected": [
                    [0, 7, 8, 3, 1, 2],
                    [0, 4, 5, 6, 1, 2],
                    [0, 9, 3, 3, 3, 1]
                ],
            }
        ),
        # Test `max_seq_len` is larger than sequence length
        #
        # Expect add pad_tk at the end of sequence, test_input will first
        # add bos and eos token.
        (
            {
                "max_seq_len": 6,
                "test_input": ['a b c', '1 2 3'],
                "expected": [[0, 4, 5, 6, 1, 2], [0, 7, 8, 3, 1, 2]],
            }
        ),
        # Test `max_seq_len` is smaller than sequence length
        #
        # Expect truncate the sequnce to `max_seq_len` in list.
        (
            {
                "max_seq_len": 3,
                "test_input": ['a b c a b c', '1 2 3 1 2 3'],
                "expected": [[0, 4, 5], [0, 7, 8]],
            }
        )
    ]
)
def test_batch_enc(tk2id, parameters):
    r"""Turn text batch to token batch"""

    tknz = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert (
        tknz.batch_enc(
            parameters['test_input'],
            max_seq_len=parameters['max_seq_len']
        )
        ==
        parameters['expected']
    )


@pytest.mark.parametrize(
    "parameters",
    [
        # Test empty input sequence
        #
        # Expect output empty sequence.
        (
            {
                "rm_sp_tks": False,
                "test_input": [[]],
                "expected": [''],
            }
        ),
        # Test ids to tokens
        #
        # Expect output tokens transferred from ids.
        (
            {
                "rm_sp_tks": False,
                "test_input": [[7, 8], [4, 5, 6], [9, 3, 3, 3]],
                "expected": ['1 2', 'a b c', '哈 [unk] [unk] [unk]'],
            }
        ),
        # Test special tokens
        #
        # Test output special tokens.
        (
            {
                "rm_sp_tks": False,
                "test_input": [[0, 1, 2, 3]],
                "expected": ['[bos] [eos] [pad] [unk]'],
            }
        ),
        # Test remove special tokens
        #
        # Expect remove special tokens except unknown tokens.
        (
            {
                "rm_sp_tks": True,
                "test_input": [[0, 1, 2, 3]],
                "expected": ['[unk]'],
            }
        ),
    ]
)
def test_batch_dec(tk2id, parameters):
    r"""Turn token batch to text batch"""

    tknz = WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=tk2id,
    )

    assert (
        tknz.batch_dec(
            parameters['test_input'],
            rm_sp_tks=parameters['rm_sp_tks']
        )
        ==
        parameters['expected']
    )
