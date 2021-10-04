r"""Test :py:mod:`lmp.util.tknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

import lmp.util.tknzr
from lmp.tknzr import BaseTknzr


def test_module_function():
    """Ensure module function's signature."""
    assert inspect.isfunction(lmp.util.tknzr.create)
    assert inspect.signature(lmp.util.tknzr.create) == Signature(
        parameters=[
            Parameter(
                name='tknzr_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
            Parameter(
                name='kwargs',
                kind=Parameter.VAR_KEYWORD,
                default=Parameter.empty,
                annotation=Optional[Dict],
            ),
        ],
        return_annotation=BaseTknzr,
    )

    assert inspect.isfunction(lmp.util.tknzr.load)
    assert inspect.signature(lmp.util.tknzr.load) == Signature(
        parameters=[
            Parameter(
                name='exp_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
            Parameter(
                name='tknzr_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=BaseTknzr,
    )
