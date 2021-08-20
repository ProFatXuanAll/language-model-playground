r"""Test the model's generation

Test target:
- :py:meth:`lmp.infer._top_1.Top1Infer.gen`.
"""
import pytest

from lmp.infer._top_1 import Top1Infer


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
        (
            # Test normal generation
            #
            # Expect only output h when max_vocab_size is 1. Output length must be 5 when 
            # input max_seq_len is 6, since there is one length for bos.
            {
                'max_seq_len': 6,
            },
            "hhhhh",
            "hhhhh",
        ),
        (
            # Test max_seq_len
            #
            # Expect redundant input text shoud be remove when max_seq_len
            # is larger than input length
            {
                'max_seq_len': 2,
            },
            "hel",
            "h",
        ),
        (
            # Test all unkown input
            #
            # Expected all unkown when input chinese characters
            {
                'max_seq_len': 5,
            },
            "你好世界",
            "[unk]" * 4,
        ),
    ]
)
def test_gen(tknzr, model, parameters, test_input, expected, reset_pad_tkid):
    r"""Test :py:meth:lmp.infer._top_1.Top1Infer.gen"""

    infer = Top1Infer(max_seq_len=parameters['max_seq_len'])

    gen = infer.gen(
        model=model,
        tknzr=tknzr,
        txt=test_input,
    )

    assert gen == expected
