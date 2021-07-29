r"""Test :py:class:`lmp.dset.ChPoemDset` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Optional

from lmp.dset.util import download, norm, pad_to_max, trunc_to_max


def test_function():
    r"""Ensure function's signature."""
    assert inspect.isfunction(download)
    assert inspect.signature(download) == Signature(
        parameters=[
            Parameter(
                name='url',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
            Parameter(
                name='file_path',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=None,
    )

    assert inspect.isfunction(norm)
    assert inspect.signature(norm) == Signature(
        parameters=[
            Parameter(
                name='txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=str,
    )
    assert inspect.isfunction(trunc_to_max)
    assert inspect.signature(trunc_to_max) == Signature(
        parameters=[
            Parameter(
                name='seq',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='max_seq_len',
                kind=Parameter.KEYWORD_ONLY,
                default=-1,
                annotation=Optional[int],
            ),
        ],
    )

    assert inspect.isfunction(pad_to_max)
    assert inspect.signature(pad_to_max) == Signature(
        parameters=[
            Parameter(
                name='seq',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='pad',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='max_seq_len',
                kind=Parameter.KEYWORD_ONLY,
                default=-1,
                annotation=Optional[int],
            ),
        ],
    )
