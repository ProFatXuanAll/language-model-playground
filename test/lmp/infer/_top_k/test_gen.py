import pytest

from lmp.infer._top_k import TopKInfer


@pytest.mark.parametrize(
    "parameters,test_input,expected",
    [
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
def test_gen(tknzr, model, parameters, test_input, expected):
    r"""Test :py:meth:lmp.infer._top_k.TopKInfer.gen"""

    infer = TopKInfer(k=3, max_seq_len=parameters['max_seq_len'])

    gen = infer.gen(
        model=model,
        tknzr=tknzr,
        txt=test_input,
    )

    assert gen == expected
